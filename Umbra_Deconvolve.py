#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Umbra Deconvolve (preview), Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
"""Umbra Deconvolve (preview): AI deconvolution for linear images. No preview: set the controls, press Apply.
Correct Only: fixes star shapes (aberrations) without sharpening. Strength: how far to sharpen toward the sampling limit; always limited by the noise.
Strength sharpens nebula/galaxy detail only (stars are only made round). Sharpen Stars is the only control that shrinks stars; Halo Reduction is the only one that pulls in star halos
(both need a model with separate star controls, n_cond >= 6).
Automatic PSF measures the star size from the image; switch it off to set the PSF diameter by hand.
Run in Siril (Scripts menu), or standalone:
  python Umbra_Deconvolve.py --image in.fits --out out.fits [--model PATH.onnx] [--strength 0..1] [--stars 0..1] [--halo 0..1] [--correct-only] [--fwhm PX] [--sharpening 0..8 (fb models)]
  python Umbra_Deconvolve.py --ui-test        (opens the window on a fake image, then exits)
  python Umbra_Deconvolve.py --check-updates | --download [MODEL_ID|all]"""
import sys, os, json, hashlib, argparse, time
from pathlib import Path
_STANDALONE = any(a in ('--image', '--ui-test', '--download', '--check-updates') or a.startswith(('--image=', '--download=')) for a in sys.argv)
if __name__ == '__main__' and not _STANDALONE and '--help' not in sys.argv and '-h' not in sys.argv:
    import sirilpy as _siril_deps
    def _ort_requirement():
        """onnxruntime variant for this machine (all variants import as 'onnxruntime'; only one may be installed). An existing one is kept."""
        import importlib.metadata as _md, platform as _pf, shutil as _sh
        for n in ('onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-rocm', 'onnxruntime'):
            try:
                _md.version(n); return n
            except Exception:
                pass
        if _pf.system() == 'Windows': return 'onnxruntime-directml'          # GPU through DirectX 12 (NVIDIA / AMD / Intel); CUDA needs onnxruntime-gpu
        if _pf.system() == 'Linux' and _sh.which('nvidia-smi'): return 'onnxruntime-gpu'   # needs the CUDA 12 + cuDNN 9 libraries; falls back to CPU without them
        return 'onnxruntime'                                                   # macOS: Core ML is built in; Linux without NVIDIA: CPU
    _siril_deps.ensure_installed('numpy', _ort_requirement(), 'PyQt6', 'scipy', 'astropy')
import numpy as np

APP_NAME = 'Umbra Deconvolve'
VERSION = '0.0.14'  # preview
# 0.0.14: in-script model download and update check. If no model is installed, Run is disabled and a 'Download model' button fetches it
#        (progress, temp file, SHA-256 check against the manifest, then moved into place). 'Check for updates' compares installed models
#        with the manifest and the script version with the latest release (it only shows a link; the script never replaces itself).
#        Nothing is downloaded until a button is pressed; only .onnx files from this project's GitHub releases are fetched, over HTTPS,
#        and no information about you or your machine is sent. Offline works with the installed models. UMBRA_DECONV_OFFLINE=1 turns
#        the manifest lookup off. Standalone: --download [ID|all], --check-updates.
# 0.0.13: runs on ONNX Runtime instead of PyTorch (no torch needed at all; one self-contained file + the .onnx models).
#        Models are umbra_deconv_*.onnx (converted from the 0.0.12 .pt models, same outputs to within 0.5 noise-sigma on a 10-40 sigma
#        change). GPU: CUDA (NVIDIA, Windows/Linux), DirectML (any DX12 GPU, Windows), ROCm (AMD, Linux), Core ML (Apple GPU), each checked
#        against the CPU on first load, otherwise CPU (UMBRA_DECONV_CPU=1 forces it). Tiles are fixed 512 px.
# 0.0.12: fine-detail models (umbra_deconv_*finedetail*.pt) get the sky mask automatically (no grain gain in empty sky);
#        optional 'Protect star shapes (slower)': stars from umbra_deconv_fb_r2_lanes.pt, detail from the fine-detail model (~2x time).
# 0.0.11: fb models: the PSF is measured ONCE and each pass uses a scheduled PSF that approaches the sampling floor (no more
#        PSF-sized blobs / beads / dark holes at high Sharpening); correlated (drizzled) noise is detected and its noise estimate
#        raised when the FITS header / history / file name says drizzle (a drizzled file without a header marker: put
#        'drizzle' in its file name); saturated star cores are kept from the original (they were erased). Same controls.
# 0.0.10: Sharpening tooltip covers umbra_deconv_fb_r1 (trained on its own repeated passes: clean to ~6, beading from ~7) and fb_a2 (beading from ~5).
# 0.0.9: starless images: if Automatic PSF reads > 9 px (star residues), a 4.0 px fallback is used and the log says so (set the PSF Diameter to the starred image's value).
# 0.0.8: fb models get ONE control, Sharpening 0-8 (steps of 0.25): 0 = Correct Only; v >= 1 -> Correct Only, floor(v) full passes,
#        then one pass at strength frac(v). Older models keep the single Strength slider (unchanged). Passes spin box removed.
# 0.0.7: Passes (1-4): Correct Only once, then N sharpen passes with the PSF re-measured each time; Strength scales the LAST pass.
#        Default 3 for 'fb' models (trained as a repeatable small step), 1 otherwise (= 0.0.6 behaviour).
# 0.0.6: Detail Boost removed again (owner: masks are hard to calibrate across images; sharpening comes from training only).
# 0.0.5: Detail Boost slider (withdrawn in 0.0.6).
# 0.0.4: works when Siril's current working folder has been deleted (torch failed to import in that case).
# 0.0.3: torch is imported once at startup; if it fails, the window shows the original error (also written to Umbra_Deconvolve_error.log).
# 0.0.2: Sharpen Stars and Halo Reduction sliders enabled (--stars / --halo on the CLI); disabled for models with n_cond < 6; Correct Only zeroes all three.
MODEL_DIR = os.environ.get('UMBRA_DECONV_MODEL_DIR') or str(Path(__file__).resolve().parent / 'Umbra Deconvolve')   # *.onnx models; override one model with $UMBRA_DECONV_MODEL
PREFS_FILE = Path(__file__).with_name('Umbra_Deconvolve_prefs.json')
LOG_FILE = Path(__file__).with_name('Umbra_Deconvolve_error.log')
_IMPORT_ERROR = None   # first onnxruntime / scipy import failure (remembered so every later attempt shows the real cause)


def _import_engine():
    """Import onnxruntime and scipy exactly once; on failure keep the ORIGINAL error (with traceback) for every later attempt."""
    global _IMPORT_ERROR
    if _IMPORT_ERROR is not None:
        raise RuntimeError(_IMPORT_ERROR)
    try:   # Siril's working folder may have been deleted; some imports then fail in os.getcwd() -> move to a folder that exists
        os.getcwd()
    except OSError:
        os.chdir(str(Path(__file__).resolve().parent))
    try:
        import onnxruntime  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        _IMPORT_ERROR = f'{type(e).__name__}: {e} (full details: {LOG_FILE})'
        try:
            LOG_FILE.write_text(f'{time.ctime()}\npython {sys.executable}\nsys.path {sys.path}\n\n{tb}')
        except Exception:
            pass
        print(tb, flush=True)
        raise RuntimeError(_IMPORT_ERROR) from e


MODEL_LABELS = {'umbra_deconvolution_1.0': 'Umbra Deconvolution 1.0',
                'umbra_deconvolution_1.1a_experimental': 'Umbra Deconvolution 1.1a experimental'}


def list_models():
    files = sorted(Path(MODEL_DIR).glob('*.onnx')) if Path(MODEL_DIR).is_dir() else []
    return [(MODEL_LABELS.get(f.stem, f.stem), str(f)) for f in files]


def default_model_path():
    env = os.environ.get('UMBRA_DECONV_MODEL')
    if env:
        return env
    try:
        saved = json.loads(PREFS_FILE.read_text()).get('model')
        if saved and saved.endswith('.onnx') and Path(saved).is_file():
            return saved
    except Exception:
        pass
    items = list_models()
    return items[0][1] if items else None


def remember_model(path):
    try:
        PREFS_FILE.write_text(json.dumps({'model': str(path)}))
    except Exception:
        pass



# ----------------------------------------------------------------------------------------------- model download / updates
REPO = 'Kishanhpatel/umbra-deconvolve'
MANIFEST_URL = f'https://raw.githubusercontent.com/{REPO}/main/models/models.json'
RELEASES_URL = f'https://github.com/{REPO}/releases'
_REDIRECT_HOSTS = {'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com', 'raw.githubusercontent.com'}
import urllib.request
USER_AGENT = 'UmbraDeconvolve'    # generic client name only; nothing about the user or the machine is sent


class DownloadError(Exception):
    pass


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlparse
        u = urlparse(newurl)
        if u.scheme != 'https' or u.hostname not in _REDIRECT_HOSTS:
            raise DownloadError(f'Refused a redirect to {u.hostname or newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _ssl_context():
    """Default certificate store; if the Python build has none (python.org macOS builds without 'Install Certificates'), use certifi's.
    Certificates are always verified."""
    import ssl
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get('x509_ca') and not ctx.cert_store_stats().get('x509'):
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    return ctx


def _open(url, timeout=20):
    """The only network entry point: HTTPS GET with a generic user agent; redirects only to GitHub hosts; certificates verified."""
    opener = urllib.request.build_opener(_SafeRedirect(), urllib.request.HTTPSHandler(context=_ssl_context()))
    return opener.open(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=timeout)


def _valid_model_url(url, file):
    from urllib.parse import urlparse
    u = urlparse(str(url))
    return u.scheme == 'https' and u.hostname == 'github.com' and u.path.startswith(f'/{REPO}/releases/download/') and u.path.endswith('/' + file) and not u.query


def _valid_entry(m):
    import re
    try:
        f = m['file']
        return (re.fullmatch(r'[A-Za-z0-9._-]{1,120}\.onnx', f) is not None and '..' not in f and _valid_model_url(m['url'], f)
                and re.fullmatch(r'[0-9a-f]{64}', str(m['sha256']).lower()) is not None and int(m['size_bytes']) > 0 and bool(m.get('name')))
    except Exception:
        return False


def parse_manifest(data):
    """Validated manifest dict {'models': [...valid entries...], 'script': {'version','url'} or None}. Bad entries are dropped."""
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('models'), list):
        raise DownloadError('The model list has an unexpected format.')
    models = [dict(m, sha256=str(m['sha256']).lower(), size_bytes=int(m['size_bytes'])) for m in data['models'] if isinstance(m, dict) and _valid_entry(m)]
    sc = data.get('script')
    if not (isinstance(sc, dict) and _version_tuple(sc.get('version')) and str(sc.get('url', '')).startswith(f'https://github.com/{REPO}/')):
        sc = None
    return dict(models=models, script=sc)


def fetch_manifest(url=None):
    try:
        with _open(url or MANIFEST_URL, 20) as resp:
            raw = resp.read(1_000_000)
        return parse_manifest(json.loads(raw.decode('utf-8')))
    except DownloadError:
        raise
    except Exception as e:
        why = ' Certificate problem: on macOS run "Install Certificates.command" for your Python.' if 'CERTIFICATE' in str(e).upper() else ''
        raise DownloadError(f'Could not reach the model list ({type(e).__name__}). Offline? The installed models still work.{why}') from e


def _version_tuple(v):
    try:
        return tuple(int(p) for p in str(v).strip().lstrip('v').split('.'))
    except Exception:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state(entry, model_dir=None):
    """'missing' | 'current' | 'update' for a manifest entry against the installed file."""
    p = Path(model_dir or MODEL_DIR) / entry['file']
    if not p.is_file():
        return 'missing'
    return 'current' if file_sha256(p) == entry['sha256'] else 'update'


def check_updates(manifest, model_dir=None):
    """-> dict(models=[(entry, state)], script_version=str|None, script_url=str|None, script_newer=bool)."""
    sc = manifest.get('script')
    newer = bool(sc) and _version_tuple(sc['version']) > (_version_tuple(VERSION) or (0,))
    return dict(models=[(m, model_state(m, model_dir)) for m in manifest['models']],
                script_version=sc['version'] if sc else None, script_url=sc['url'] if sc else None, script_newer=newer)


def download_model(entry, model_dir=None, progress=None, cancel=None):
    """Download one manifest entry into the models folder: temp file, size and SHA-256 check, then moved into place.  Returns the path."""
    if not _valid_entry(entry):
        raise DownloadError('Refused: not a valid model entry.')
    d = Path(model_dir or MODEL_DIR)
    try:
        d.mkdir(parents=True, exist_ok=True)
        free = __import__('shutil').disk_usage(d).free
    except OSError as e:
        raise DownloadError(f'Cannot write to the models folder {d}: {e}') from e
    size = int(entry['size_bytes'])
    if free < size * 1.1:
        raise DownloadError(f'Not enough free disk space in {d} ({size / 1e6:.0f} MB needed).')
    part, final = d / f'.{entry["file"]}.part', d / entry['file']
    h = hashlib.sha256(); got = 0
    try:
        with _open(entry['url'], 30) as resp, open(part, 'wb') as out:
            while True:
                if cancel and cancel():
                    raise DownloadError('Download cancelled.')
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if got > size * 1.01 + 1024:
                    raise DownloadError('The download is larger than expected; stopped.')
                out.write(chunk); h.update(chunk)
                if progress: progress(got, size)
        if got != size:
            raise DownloadError(f'Incomplete download ({got} of {size} bytes).')
        if h.hexdigest() != entry['sha256']:
            raise DownloadError('Checksum mismatch: the file was not installed. Try again later.')
        os.replace(part, final)
        return final
    except DownloadError:
        raise
    except Exception as e:
        raise DownloadError(f'Download failed ({type(e).__name__}: {e}).') from e
    finally:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass


def offline_mode():
    return bool(os.environ.get('UMBRA_DECONV_OFFLINE'))


def describe_updates(res):
    """Plain-text summary lines for a check_updates() result."""
    lines = []
    if res['script_newer']:
        lines.append(f"A newer script is available: {res['script_version']} (you have {VERSION}). {res['script_url']}")
    elif res['script_version']:
        lines.append(f'The script is up to date ({VERSION}).')
    for m, st in res['models']:
        if st == 'missing':
            lines.append(f"Available to download: {m['name']} ({m['size_bytes'] / 1e6:.0f} MB).")
        elif st == 'update':
            lines.append(f"Update available for {m['name']} (the installed file differs from the published one).")
    if not any(st != 'current' for _, st in res['models']):
        lines.append('Installed models are up to date.')
    return lines


# ----------------------------------------------------------------------------------------------- engine
import math
from scipy.ndimage import (gaussian_filter, maximum_filter, uniform_filter, correlate1d, label, find_objects, binary_dilation)

TILE = 512
OVERLAP = 64
K_COMP = 8.0
AUTO_PSF_MAX = 9.0       # larger Automatic PSF readings come from star residues (starless images)
STARLESS_DENSITY = 15.0  # isolated stars per megapixel below which the image counts as starless
STARLESS_FWHM = 4.0      # fallback used then
SCHED_R = 0.8
SCHED_FLOOR = 1.0
CORR_REF = 2.2
CORR_TRIGGER = 2.8
CORR_C2R = 1.64


# ------------------------------------------------------------------------------------------------ helpers
def lmedian(a, axis=None):
    """torch.median semantics: the LOWER of the two middle values for an even count (np.median would average them)."""
    a = np.asarray(a)
    if axis is None:
        a = a.reshape(-1); axis = 0
    n = a.shape[axis]
    k = (n - 1) // 2
    return np.take(np.partition(a, k, axis=axis), k, axis=axis)


def _resize_matrix(n_out, n_in):
    """Bilinear (align_corners=False, torch semantics) as an (n_out, n_in) matrix."""
    src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = (src - i0).astype(np.float32)
    M = np.zeros((n_out, n_in), np.float32)
    M[np.arange(n_out), i0] += 1 - w
    M[np.arange(n_out), i1] += w
    return M


def resize_bilinear(a, H, W):
    """(C,h,w) -> (C,H,W), same as F.interpolate(mode='bilinear', align_corners=False)."""
    Ay = _resize_matrix(H, a.shape[1]); Ax = _resize_matrix(W, a.shape[2])
    return np.einsum('Hh,chw,Ww->cHW', Ay, a.astype(np.float32), Ax, optimize=True).astype(np.float32)


def _sub(x, max_px=4_000_000):
    st = max(1, int((x.shape[-1] * x.shape[-2] / max_px) ** 0.5))
    return x[:, ::st, ::st]


def haar_sigma(x):
    x = np.asarray(x, np.float32)
    x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2]
    a, b, c, d_ = x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]
    d = (a - b - c + d_) / 2.0
    ok = np.isfinite(a) & np.isfinite(d_) & (a != 0) & np.isfinite(d)
    out = []
    for ch in range(x.shape[0]):
        v = _sub(d[ch:ch + 1])[_sub(ok[ch:ch + 1])]
        out.append(np.float32(1.4826) * lmedian(np.abs(v - lmedian(v))) if v.size > 100 else np.float32(0.0))
    return np.array(out, np.float32)


def image_stats(x):
    """(C,H,W) -> (m (C,), s (C,)): per-channel median offset and noise sigma (floored)."""
    x = np.asarray(x, np.float32)
    xs = _sub(x)
    m = np.array([lmedian(np.nan_to_num(xs[c][np.isfinite(xs[c])])) for c in range(x.shape[0])], np.float32)
    s = haar_sigma(x)
    rng = np.array([np.quantile(np.nan_to_num(xs[c]).ravel(), 0.999) if xs[c].size < 16_000_000 else xs[c].max() for c in range(x.shape[0])], np.float32) - m
    return m, np.maximum(s, 1e-5 * np.clip(np.abs(rng), 1e-12, None)).astype(np.float32)


def compress(xn):
    return (K_COMP * np.arcsinh(xn / K_COMP)).astype(np.float32)


def decompress(y):
    return (K_COMP * np.sinh(np.clip(y, -150, 150) / K_COMP)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ Automatic PSF
def _interp4(k):
    """(N,K,K) -> (N,4K,4K), bilinear x4 (align_corners=False)."""
    M = _resize_matrix(4 * k.shape[-1], k.shape[-1])
    return np.einsum('ak,nkl,bl->nab', M, k, M, optimize=True)


def eq_fwhm(k):
    u = _interp4(np.asarray(k, np.float32))
    pk = u.reshape(u.shape[0], -1).max(1)
    a = (u >= 0.5 * pk[:, None, None]).sum((1, 2)).astype(np.float32) / 16.0
    return 2 * np.sqrt(a / math.pi)


def shape_e(k, fw):
    N, K, _ = k.shape
    r = np.arange(K, dtype=np.float32) - K // 2
    yy, xx = np.meshgrid(r, r, indexing='ij')
    w = np.clip(k, 0, None)
    t = np.clip(w.sum((1, 2)), 1e-12, None)
    my = (w * yy).sum((1, 2)) / t; mx = (w * xx).sum((1, 2)) / t
    dy, dx = yy[None] - my[:, None, None], xx[None] - mx[:, None, None]
    w = w * (((dy * dy + dx * dx) <= (1.5 * fw)[:, None, None] ** 2).astype(np.float32))
    t = np.clip(w.sum((1, 2)), 1e-12, None)
    cyy = (w * dy * dy).sum((1, 2)) / t; cxx = (w * dx * dx).sum((1, 2)) / t; cxy = (w * dy * dx).sum((1, 2)) / t
    tr = np.clip(cxx + cyy, 1e-9, None)
    return (cxx - cyy) / tr, 2 * cxy / tr


def _blur_reflect(x, sg):
    r = int(3 * sg) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / sg) ** 2).astype(np.float32); k /= k.sum()
    p = np.pad(x, ((0, 0), (r, r)), mode='reflect')
    p = correlate1d(p, k, axis=1, mode='constant')[:, r:-r]
    p = np.pad(p, ((r, r), (0, 0)), mode='reflect')
    return correlate1d(p, k, axis=0, mode='constant')[r:-r]


def _shift_stamps(st, cx, cy, h):
    """F.grid_sample(bilinear, zeros, align_corners=True) of centred stamps re-centred by (cx, cy): out[i,j] = st[i+cy, j+cx] (zero outside)."""
    N, S, _ = st.shape
    ii = np.arange(S, dtype=np.float32)
    y = ii[None, :, None] + cy[:, None, None]; x = ii[None, None, :] + cx[:, None, None]
    y = np.broadcast_to(y, (N, S, S)); x = np.broadcast_to(x, (N, S, S))
    y0 = np.floor(y).astype(np.int64); x0 = np.floor(x).astype(np.int64)
    wy = (y - y0).astype(np.float32); wx = (x - x0).astype(np.float32)
    n = np.arange(N)[:, None, None]
    def g(yi, xi):
        ok = (yi >= 0) & (yi < S) & (xi >= 0) & (xi < S)
        return np.where(ok, st[n, np.clip(yi, 0, S - 1), np.clip(xi, 0, S - 1)], 0.0)
    return (g(y0, x0) * (1 - wy) * (1 - wx) + g(y0, x0 + 1) * (1 - wy) * wx + g(y0 + 1, x0) * wy * (1 - wx) + g(y0 + 1, x0 + 1) * wy * wx).astype(np.float32)


def measure_psf(Y, region=512, stamp=15, min_stars=8):
    """Y (3,H,W) linear.  Returns dict(fmap (H,W), global, grid, shape (2,H,W), n_stars) like udc.infer.measure_psf."""
    H, W = Y.shape[-2:]
    m, s = image_stats(Y)
    L = ((Y - m.reshape(-1, 1, 1)) / s.reshape(-1, 1, 1)).mean(0).astype(np.float32)
    d = _blur_reflect(L, 1.0) - _blur_reflect(L, 6.0)
    mx = float(L.max())
    pk = (d == maximum_filter(d, size=11, mode='constant', cval=-np.inf)) & (d > 20) & (L < 0.5 * mx)
    h = stamp // 2
    pk[:h + 2] = False; pk[-h - 2:] = False; pk[:, :h + 2] = False; pk[:, -h - 2:] = False
    ys, xs = np.nonzero(pk)
    if ys.size == 0:
        return dict(fmap=np.full((H, W), 3.0, np.float32), gl=3.0, grid=None, shape=np.zeros((2, H, W), np.float32), n=0)
    cnt = uniform_filter(pk.astype(np.float64), size=stamp, mode='constant') * stamp * stamp
    iso = cnt[ys, xs] < 1.5
    ys, xs = ys[iso], xs[iso]
    n_iso = int(ys.size)
    off = np.arange(-h, h + 1)
    ix = (ys[:, None, None] + off[None, :, None], xs[:, None, None] + off[None, None, :])
    st = L[ix]
    edge = lmedian(np.concatenate([st[:, 0], st[:, -1], st[:, :, 0], st[:, :, -1]], 1), 1)
    st = np.clip(st - edge[:, None, None], 0, None)
    st = st / np.clip(st.sum((1, 2), keepdims=True), 1e-9, None)
    oy, ox = np.meshgrid(off.astype(np.float32), off.astype(np.float32), indexing='ij')
    cy = (st * oy).sum((1, 2)); cx = (st * ox).sum((1, 2))
    st = _shift_stamps(st, cx, cy, h)
    Yn = (Y - m.reshape(-1, 1, 1)) / s.reshape(-1, 1, 1)
    stc = []
    for c in range(Yn.shape[0]):
        a = Yn[c][ix]
        ed = lmedian(np.concatenate([a[:, 0], a[:, -1], a[:, :, 0], a[:, :, -1]], 1), 1)
        a = np.clip(a - ed[:, None, None], 0, None); a = a / np.clip(a.sum((1, 2), keepdims=True), 1e-9, None)
        cyc = (a * oy).sum((1, 2)); cxc = (a * ox).sum((1, 2))
        stc.append(_shift_stamps(a, cxc, cyc, h))

    def fw(sel):
        if sel.sum() < min_stars:
            return None
        k = lmedian(st[sel], 0)[None]
        f = eq_fwhm(k)
        es = []
        for sc in stc:
            med = lmedian(sc[sel], 0)
            es.append(shape_e(med[None] / med.sum(), f))
        return float(f[0]), float(sum(e[0][0] for e in es) / len(es)), float(sum(e[1][0] for e in es) / len(es))

    g0 = fw(np.ones_like(ys, dtype=bool)) or (3.0, 0.0, 0.0)
    gl = g0[0]
    ny, nx = max(1, round(H / region)), max(1, round(W / region))
    grid = np.tile(np.array(g0, np.float32).reshape(3, 1, 1), (1, ny, nx))
    for i in range(ny):
        for j in range(nx):
            sel = (ys * ny // H == i) & (xs * nx // W == j)
            v = fw(sel)
            if v is not None:
                grid[:, i, j] = v
    maps = resize_bilinear(grid, H, W)
    return dict(fmap=maps[0], gl=gl, grid=grid[0], shape=maps[1:], n=n_iso)


def auto_psf(Y0, fwhm=None):
    """Automatic PSF (with the starless guard) or manual FWHM -> (fmap, global, grid, emap, psf_note)."""
    p = measure_psf(Y0)
    fmap, gl, grid, emap = p['fmap'], p['gl'], p['grid'], p['shape']
    psf_note = None
    dens = p['n'] / max(Y0.shape[-1] * Y0.shape[-2] / 1e6, 1e-6)
    if fwhm is None and (gl > AUTO_PSF_MAX or dens < STARLESS_DENSITY):
        psf_note = (f"Auto PSF: image looks starless ({dens:.0f} isolated stars per MP, measured {gl:.1f} px). Using {STARLESS_FWHM:.1f} px; "
                    "for starless images set the PSF Diameter to the value measured on the starred version.")
        fmap, gl, grid = np.full(Y0.shape[-2:], STARLESS_FWHM, np.float32), STARLESS_FWHM, None
        emap = np.zeros_like(emap)
    if fwhm is not None:
        fmap, gl, grid = np.full(Y0.shape[-2:], float(fwhm), np.float32), float(fwhm), None
    return fmap, gl, grid, emap, psf_note


# ------------------------------------------------------------------------------------------------ tiled network
def _window(n, ov):
    w = np.ones(n, np.float32)
    if ov > 0:
        r = np.linspace(0, 1, ov + 2, dtype=np.float32)[1:-1]; r = 0.5 - 0.5 * np.cos(np.pi * r)
        w[:ov] = r; w[-ov:] = r[::-1]
    return w


def deconvolve_once(net, y, strength=1.0, fwhm=None, progress=None, star=0.0, halo=0.0, psf_fixed=None, sigma_k=1.0):
    """net: callable (x (1,3,T,T), cond (1,6,T,T)) -> r (1,3,T,T), T = 512.  y (3,H,W) float32 linear."""
    Y = np.ascontiguousarray(y, np.float32)
    fin = np.isfinite(Y); Y0 = np.nan_to_num(Y)
    if psf_fixed is not None:
        fmap, gl, emap = psf_fixed; grid = None; psf_note = None
    else:
        fmap, gl, grid, emap, psf_note = auto_psf(Y0, fwhm)
    m, s = image_stats(Y0); m = m.reshape(-1, 1, 1); s = s.reshape(-1, 1, 1) * np.float32(sigma_k)
    xn = ((Y0 - m) / s).astype(np.float32)
    C, H, W = xn.shape
    T, ov = TILE, OVERLAP
    st = T - ov
    ny = max(1, math.ceil((H - ov) / st)) if H > T else 1
    nx = max(1, math.ceil((W - ov) / st)) if W > T else 1
    Hp, Wp = max(H, (ny - 1) * st + T), max(W, (nx - 1) * st + T)
    mode = 'reflect' if (Hp - H < H and Wp - W < W) else 'edge'
    xp = np.pad(xn, ((0, 0), (0, Hp - H), (0, Wp - W)), mode=mode)
    cond = np.stack([np.full((H, W), float(strength), np.float32), fmap / 4.0, 3 * emap[0], 3 * emap[1],
                     np.full((H, W), float(star), np.float32), np.full((H, W), float(halo), np.float32)]).astype(np.float32)
    cp = np.pad(cond, ((0, 0), (0, Hp - H), (0, Wp - W)), mode='edge')
    xc = compress(xp)
    acc = np.zeros((C, Hp, Wp), np.float32); wsum = np.zeros((1, Hp, Wp), np.float32)
    w2 = (_window(T, ov if ny > 1 else 0)[:, None] * _window(T, ov if nx > 1 else 0)[None, :])[None]
    n, tot = 0, ny * nx
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * st, ix * st
            r = net(np.ascontiguousarray(xc[None, :, y0:y0 + T, x0:x0 + T]), np.ascontiguousarray(cp[None, :, y0:y0 + T, x0:x0 + T]))[0]
            acc[:, y0:y0 + T, x0:x0 + T] += r * w2; wsum[:, y0:y0 + T, x0:x0 + T] += w2
            n += 1
            if progress: progress(n, tot)
    r = (acc / np.clip(wsum, 1e-8, None))[:, :H, :W]
    out = decompress(compress(xn) + r) * s + m
    return np.where(fin, out, Y).astype(np.float32), dict(fwhm_global=gl, fwhm_grid=grid, psf_note=psf_note)


def deconvolve_chw(net, y, strength=1.0, fwhm=None, progress=None, star=0.0, halo=0.0, two_pass=True):
    """Correct Only first, then the requested sharpening (single-strength models / Sharpening 0)."""
    if two_pass and (strength > 0 or star > 0 or halo > 0):
        yc, _ = deconvolve_once(net, y, 0.0, fwhm, None, 0.0, 0.0)
        return deconvolve_once(net, yc, strength, fwhm, progress, star, halo)
    return deconvolve_once(net, y, strength, fwhm, progress, star, halo)


# ------------------------------------------------------------------------------------------------ guards / masks
def noise_corr_factor(y, b=64):
    L = np.nan_to_num(np.asarray(y, np.float32)).mean(0)
    _, s = image_stats(np.nan_to_num(np.asarray(y, np.float32))); sh = float(s.mean())
    H, W = L.shape
    if H < 2 * b or W < 2 * b or not sh > 0:
        return 1.0, float('nan')
    blocks = sorted((float(L[yy:yy + b, xx:xx + b].std()), yy, xx) for yy in range(0, H - b + 1, b) for xx in range(0, W - b + 1, b))
    mad = lambda a: 1.4826 * float(np.median(np.abs(a - np.median(a))))
    r, r2 = [], []
    for _, yy, xx in blocks[:max(8, len(blocks) // 20)]:
        c0 = L[yy:yy + b, xx:xx + b]; c = c0 - gaussian_filter(c0, 12); q = c.reshape(b // 4, 4, b // 4, 4).mean((1, 3))
        r.append(mad(q) * 4 / sh)
        h = c0 - gaussian_filter(c0, 6); r2.append(mad(h.reshape(b // 2, 2, b // 2, 2).mean((1, 3))) * 2 / max(mad(h), 1e-30))
    c4, c2r = float(np.median(r)), float(np.median(r2))
    noise_corr_factor.last = dict(c4=c4, c2r=c2r)
    return (float(min(2.0, c4 / CORR_REF)) if (c4 > CORR_TRIGGER and c2r >= CORR_C2R) else 1.0), c4


def saturation_mask(y, fwhm):
    Y = np.nan_to_num(np.asarray(y, np.float32)); Lm = Y.max(0); gmax = float(Lm.max())
    if not gmax > 0: return None
    med = float(np.median(Lm)); _, s = image_stats(Y); sg = float(s.mean())
    Lm = Lm - med; gmax = gmax - med
    if not gmax > 0: return None
    Mx = maximum_filter(Lm, 7)
    cand = (Lm >= 0.97 * Mx) & (Mx > 0.25 * gmax) & (Mx > 50 * sg)
    lb, n = label(cand)
    if n == 0: return None
    area = np.bincount(lb.ravel()); amin = max(5.0, 0.15 * fwhm * fwhm)
    keep = np.zeros(n + 1, bool); H, W = Lm.shape; tops = []
    for k, sl in enumerate(find_objects(lb), 1):
        if sl is None or area[k] < amin: continue
        comp = lb[sl] == k; v = Lm[sl]; top = comp & (v >= 0.97 * float(v[comp].max()))
        if top.sum() < amin: continue
        yy, xx = np.nonzero(top); cy, cx = yy.mean() + sl[0].start, xx.mean() + sl[1].start
        rp = math.sqrt(top.sum() / math.pi); rr = rp + fwhm; pk = float(v[top].mean())
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ry = np.clip(np.round(cy + rr * np.sin(ang)).astype(int), 0, H - 1); rx = np.clip(np.round(cx + rr * np.cos(ang)).astype(int), 0, W - 1)
        if float(np.median(Lm[ry, rx])) < 0.6 * pk: keep[k] = True; tops.append((sl, top))
    sat = np.zeros_like(Lm, bool)
    for sl, top in tops: sat[sl] |= top
    if not sat.any(): return None
    m = binary_dilation(sat, iterations=max(2, int(round(2 * fwhm)))).astype(np.float32)
    return np.clip(gaussian_filter(m, max(1.0, fwhm)) * 1.5, 0, 1)


def star_mask(y, fwhm):
    Y = np.nan_to_num(np.asarray(y, np.float32)); L = Y.mean(0)
    _, s = image_stats(Y); sg = float(s.mean())
    sb = max(0.8, fwhm / 2.355); D = gaussian_filter(L, sb * 0.7) - gaussian_filter(L, 3 * sb)
    pk = (D == maximum_filter(D, size=int(2 * fwhm) | 1)) & (D > 25 * sg)
    ys, xs = np.nonzero(pk); H, W = L.shape; M = np.zeros((H, W), np.float32)
    bgl = gaussian_filter(L, 4 * fwhm)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for y0, x0 in zip(ys, xs):
        ry = np.clip(np.round(y0 + 2 * fwhm * np.sin(ang)).astype(int), 0, H - 1); rx = np.clip(np.round(x0 + 2 * fwhm * np.cos(ang)).astype(int), 0, W - 1)
        c = L[y0, x0] - bgl[y0, x0]; ring = float(np.median(L[ry, rx] - bgl[ry, rx]))
        if c <= 0 or ring > c / 3: continue
        r = min(4.0, 2.0 + 0.6 * math.log10(max(D[y0, x0] / (25 * sg), 1.0))) * fwhm; R = int(r + 2 * fwhm)
        ya, yb, xa, xb = max(0, y0 - R), min(H, y0 + R + 1), max(0, x0 - R), min(W, x0 + R + 1)
        yy, xx = np.mgrid[ya:yb, xa:xb]; d = np.hypot(yy - y0, xx - x0)
        M[ya:yb, xa:xb] = np.maximum(M[ya:yb, xa:xb], np.clip((r - d) / (0.7 * fwhm) + 0.5, 0, 1))
    return gaussian_filter(M, 0.5 * fwhm).clip(0, 1)


def background_weight(y, fwhm):
    Y = np.nan_to_num(np.asarray(y, np.float32)); L = Y.mean(0)
    _, s = image_stats(Y); sg = float(s.mean())
    sky = float(np.percentile(gaussian_filter(L, 4)[::4, ::4], 10))
    sm = max(1.0, 0.7 * fwhm); snr = (gaussian_filter(L, sm) - sky) / (sg / max(1.0, 2 * math.sqrt(math.pi) * sm))
    w = np.clip((snr - 1.5) / 3.5, 0, 1)
    return gaussian_filter(w, sm).astype(np.float32)


# ------------------------------------------------------------------------------------------------ scheduled passes
def deconvolve_scheduled(net, y, plan, fwhm=None, progress=None, star=0.0, halo=0.0, r=None, floor=None, sigma_fix=True,
                         sat_protect=True, drizzled=None, bg_protect=False, net_detail=None, blend='pass'):
    r = SCHED_R if r is None else r; floor = SCHED_FLOOR if floor is None else floor
    y = np.ascontiguousarray(y, np.float32)
    fmap, g0, grid, emap, psf_note = auto_psf(np.nan_to_num(y), fwhm)
    g0 = float(g0)
    sk, c4 = noise_corr_factor(y) if sigma_fix else (1.0, float('nan'))
    if sigma_fix and drizzled is True: sk = float(np.clip(c4 / CORR_REF, 1.3, 2.0)) if c4 == c4 else 1.3
    else: sk = 1.0
    P = len(plan)
    def pr(k):
        return (lambda n, t: progress(k, n, t)) if progress else None
    z, _ = deconvolve_once(net, y, 0.0, None, pr(0), psf_fixed=(fmap, g0, emap), sigma_k=sk)
    fws = []
    smk = star_mask(y, g0) if net_detail is not None else None
    bwd = background_weight(y, g0) if net_detail is not None else None
    zd = z
    for k, st in enumerate(plan, 1):
        fk = floor + (g0 - floor) * r ** (k - 1) if g0 > floor else g0
        sc = fk / g0; last = k == P; kw = dict(psf_fixed=(fmap * sc, fk, emap), sigma_k=sk)
        zin = z
        z, _ = deconvolve_once(net, zin, float(st), None, pr(k), star if last else 0.0, halo if last else 0.0, **kw)
        if net_detail is not None:
            src = zin if blend == 'pass' else zd
            zd, _ = deconvolve_once(net_detail, src, float(st), None, None, star if last else 0.0, halo if last else 0.0, **kw)
            zd = src + bwd * (zd - src)
            if blend == 'pass': z = smk * z + (1 - smk) * zd; zd = z
        fws.append(round(fk, 3))
    if net_detail is not None and blend != 'pass': z = (smk * z + (1 - smk) * zd).astype(np.float32)
    sat_px = 0
    mk = saturation_mask(y, g0) if (sat_protect and P) else None
    if mk is not None:
        sat_px = int((mk > 0.5).sum())
        z = np.where(np.isfinite(y), z * (1 - mk) + np.nan_to_num(y) * mk, y).astype(np.float32)
    if bg_protect and P:
        bw = background_weight(y, g0)
        z = np.where(np.isfinite(y), np.nan_to_num(y) + bw * (z - np.nan_to_num(y)), y).astype(np.float32)
    return z, dict(fwhm_global=g0, fwhm_grid=grid, psf_note=psf_note, fwhm_per_pass=fws, sigma_k=sk, noise_corr=c4, sat_px=sat_px)


# ------------------------------------------------------------------------------------------------ ONNX Runtime session
class OnnxNet:
    """Fixed 512 x 512 UmbraNet in ONNX Runtime.  GPU providers are tried in order of preference for the platform (CUDA, DirectML,
    ROCm, Core ML); each is checked against the CPU provider on one probe tile (fp32 results must agree to 1e-3 relative) and the
    first that works is used.  Anything missing, failing or disagreeing falls through to the next, and finally to the CPU.
    Set UMBRA_DECONV_CPU=1 to force the CPU.  self.provider names what is running; self.note says why a GPU was not used."""

    def __init__(self, path, prefer_gpu=True):
        import onnxruntime as ort
        self.path = str(path); self.provider = 'CPU'; self.note = None
        so = ort.SessionOptions(); so.log_severity_level = 3
        cpu_holder = []
        def mk_cpu():
            if not cpu_holder:
                cpu_holder.append(ort.InferenceSession(self.path, so, providers=['CPUExecutionProvider']))
            return cpu_holder[0]
        self.sess = None
        if os.environ.get('UMBRA_DECONV_CPU'):
            prefer_gpu = False
        if prefer_gpu:
            avail = ort.get_available_providers()
            cands = [  # (provider name, label, options, session options tweaks)
                ('CUDAExecutionProvider', 'CUDA', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}, {}),
                ('DmlExecutionProvider', 'DirectML', {}, {'dml': True}),
                ('ROCMExecutionProvider', 'ROCm', {}, {}),
                ('CoreMLExecutionProvider', 'CoreML (GPU)', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}, {}),
            ]
            notes = []
            rng = np.random.default_rng(1)
            x = (rng.standard_normal((1, 3, TILE, TILE)) * 2).astype(np.float32); c = rng.random((1, 6, TILE, TILE)).astype(np.float32)
            for name, label, opts, tw in cands:
                if name not in avail:
                    continue
                try:
                    so2 = ort.SessionOptions(); so2.log_severity_level = 3
                    if tw.get('dml'):                                  # DirectML requirements
                        so2.enable_mem_pattern = False; so2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    g = ort.InferenceSession(self.path, so2, providers=[(name, opts), 'CPUExecutionProvider'])
                    if name not in g.get_providers():
                        raise RuntimeError('provider did not load')
                    a = g.run(None, {'x': x, 'cond': c})[0]
                    b = mk_cpu().run(None, {'x': x, 'cond': c})[0]
                    err = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))
                    if np.isfinite(err) and err < 1e-3:
                        self.sess, self.provider = g, label
                        break
                    notes.append(f'{label} result differed from the CPU ({err:.1e})')
                except Exception as e:
                    notes.append(f'{label} unavailable ({str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__})')
            if self.sess is None and notes:
                self.note = '; '.join(notes) + '; using the CPU.'
            elif self.sess is None:
                self.note = 'No GPU provider in this onnxruntime build; using the CPU.'
        if self.sess is None:
            self.sess = mk_cpu()

    def __call__(self, x, cond):
        return self.sess.run(None, {'x': x, 'cond': cond})[0]


# ----------------------------------------------------------------------------------------------- script glue
SHARP_DEFAULT = 3.0


def is_fb(model_path):
    """fb models are trained as a small repeatable step (0.65x blur per pass) and use the Sharpening control (0-8 passes)."""
    st = Path(str(model_path or '')).stem.lower()
    return st.startswith('umbra_deconv_fb') or st.startswith('umbra_deconv_fs') or st.startswith('umbra_deconvolution_')


STAR_MODEL_NAME = 'umbra_deconvolution_1.0.onnx'   # 0.0.12: star shapes for 'Protect star shapes'


def is_finedetail(model_path):
    st = Path(str(model_path or '')).stem.lower()
    return 'finedetail' in st or st.startswith('umbra_deconvolution_1.1')


def pass_plan(v):
    """Sharpening v (0..8, steps of 0.25) -> list of per-pass strengths after Correct Only: floor(v) x 1.0, then frac(v) if > 0."""
    v = max(0.0, min(8.0, round(float(v) * 4) / 4))
    n, f = int(v), v - int(v)
    return [1.0] * n + ([round(f, 4)] if f > 1e-6 else [])


def deconvolve_array(model, data, strength, fwhm, progress=None, star=0.0, halo=0.0, sharpening=None, drizzled=None, finedetail=False, star_model=None):
    """data HxW, CxHxW (C = 1 or 3) or HxWxC (C = 1 or 3) float32 -> (same shape, info). Mono is replicated to 3 channels and averaged back.
    sharpening None (older models): Correct Only + one sharpen pass at `strength` (0.0.6 behaviour).
    sharpening v (fb models; `strength` ignored): Correct Only once, then the passes of pass_plan(v) (Sharpen Stars / Halo Reduction on the
    last pass); 0.0.11: the PSF is measured once and scheduled per pass (udc.infer.deconvolve_scheduled); every pass is tiled like a single pass (same memory)."""
    data = np.asarray(data, np.float32)
    layout = 'hw'
    if data.ndim == 3:
        if data.shape[0] in (1, 3):
            layout = 'chw'
        elif data.shape[2] in (1, 3):
            layout, data = 'hwc', np.ascontiguousarray(data.transpose(2, 0, 1))
        else:
            raise ValueError(f'Unsupported image shape {data.shape}')
    chw = data[None] if data.ndim == 2 else data
    mono = chw.shape[0] == 1
    y = np.ascontiguousarray(np.repeat(chw, 3, 0) if mono else chw, np.float32)
    fw = None if fwhm is None else float(fwhm)
    plan = None if sharpening is None else pass_plan(sharpening)
    if plan is None or not plan:
        out, info = deconvolve_chw(model, y, strength=float(strength) if plan is None else 0.0, fwhm=fw, progress=progress,
                                   star=float(star), halo=float(halo))           # older models, or Sharpening 0 (= Correct Only)
    else:
        P = len(plan)
        def prog(k, n, t):                                                 # progress per pass: (pass, tile) over Correct Only + P passes
            if progress: progress(k * t + n, (P + 1) * t)
        if star_model is not None:                                         # 0.0.12 Protect star shapes: stars from star_model, rest from the fine-detail model
            out, info = deconvolve_scheduled(star_model, y, plan, fw, progress=prog, star=float(star), halo=float(halo), drizzled=drizzled,
                                             net_detail=model, blend='end')
        else:
            out, info = deconvolve_scheduled(model, y, plan, fw, progress=prog, star=float(star), halo=float(halo), drizzled=drizzled,
                                             bg_protect=bool(finedetail))   # 0.0.11 / 0.0.12 (sky mask for fine-detail models)
        notes = [info.get('psf_note')] if info.get('psf_note') else []
        if info.get('sigma_k', 1.0) > 1.0:
            notes.append(f"Drizzled stack: noise estimate x{info['sigma_k']:.2f} so it is not sharpened into beads.")
        if info.get('sat_px', 0) > 0:
            notes.append('Saturated star cores kept as in the original (they cannot be deconvolved).')
        info['psf_note'] = ' '.join(notes) if notes else None
    out = np.asarray(out, np.float32)
    if mono:
        out = out.mean(0, keepdims=True)
    if layout == 'hw':
        out = out[0]
    elif layout == 'hwc':
        out = np.ascontiguousarray(out.transpose(1, 2, 0))
    return out, info


class Engine:
    """Loads the model once (cached)."""

    def __init__(self, model_path=None):
        _import_engine()
        path = Path(model_path or default_model_path() or '')
        if not path.is_file():
            raise FileNotFoundError(f'Model not found: {path} (put a *.onnx in {MODEL_DIR} or set UMBRA_DECONV_MODEL)')
        self.model = OnnxNet(path)
        self.device = self.model.provider
        self.model_path = str(path)
        self.last_fwhm = None
        self.last_note = None
        self.n_cond = 6
        self.has_star_controls = True

    def run(self, data, strength, fwhm, progress=None, star=0.0, halo=0.0, sharpening=None, drizzled=None, protect_stars=False):
        data = np.ascontiguousarray(data, np.float32)
        fd = is_finedetail(self.model_path); sm = None
        if fd and protect_stars and sharpening:
            sp = Path(MODEL_DIR) / STAR_MODEL_NAME
            if not sp.is_file():
                raise FileNotFoundError(f'Protect star shapes needs {sp}')
            sm = get_engine(str(sp)).model
        out, info = deconvolve_array(self.model, data, strength, fwhm, progress, star, halo, sharpening, drizzled, finedetail=fd, star_model=sm)
        g = info.get('fwhm_global')
        self.last_fwhm = None if g is None else float(g)
        notes = [f'Ran on {self.device}.'] + ([self.model.note] if self.model.note else []) + ([info['psf_note']] if info.get('psf_note') else [])
        self.last_note = ' '.join(notes)
        return out


_ENGINE = {}


def get_engine(model_path=None):
    key = str(model_path or default_model_path())
    if key not in _ENGINE:
        _ENGINE[key] = Engine(key)
    return _ENGINE[key]


def stretched_warning(data):
    """Message if the data looks stretched (median > 0.2 of max), else None."""
    d = np.asarray(data, np.float32)
    fin = d[np.isfinite(d)]
    if fin.size == 0:
        return None
    mx = float(fin.max())
    if mx > 0 and float(np.median(fin)) > 0.2 * mx:
        return 'the data looks stretched (median above 20% of the maximum); this tool expects linear data. Running anyway.'
    return None


# ----------------------------------------------------------------------------------------------- Siril I/O
def is_drizzled(header_text=None, history=None):
    """0.0.11: drizzled stacks have spatially correlated noise -> the engine raises its noise estimate (otherwise it sharpens the
    noise into beads / pits).  Detected from the FITS header / history text only (image statistics false-trigger on star fields)."""
    t = ' '.join([str(header_text or '')] + [str(h) for h in (history or [])]).lower()
    return 'drizzle' in t


def siril_drizzled(siril):
    try:
        hdr = siril.get_image_fits_header(return_as='str')
    except Exception:
        hdr = None
    try:
        hist = siril.get_image_history()
    except Exception:
        hist = None
    try:
        fn = siril.get_image_filename()
    except Exception:
        fn = None
    return is_drizzled(f'{hdr} {fn}', hist)


def read_pixels(siril):
    data = siril.get_image_pixeldata()
    if data is None:
        fit = siril.get_image(with_pixels=True)
        data = fit.data if fit is not None else None
    if data is None:
        raise RuntimeError('No image loaded in Siril.')
    return np.array(data, copy=True)


def decode_pixels(raw):
    """Siril pixels -> float32 HxW or CxHxW (uint16 scaled to 0..1)."""
    raw = np.asarray(raw)
    if raw.dtype not in (np.dtype('float32'), np.dtype('uint16')):
        raise ValueError('Expected Siril float32 or uint16 pixels.')
    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[0] in (1, 3)):
        data = raw.astype(np.float32)
    else:
        raise ValueError('Only monochrome or channel-first RGB images are supported.')
    if raw.dtype == np.uint16:
        data /= 65535.0
    return np.ascontiguousarray(data), dict(dtype=raw.dtype, shape=raw.shape)


def encode_pixels(data, meta):
    if meta['dtype'] == np.uint16:
        out = np.rint(np.clip(data, 0, 1) * 65535).astype(np.uint16)
    else:
        out = np.asarray(data, np.float32)
    if out.shape != meta['shape']:
        raise ValueError('Output shape does not match the captured image.')
    return np.ascontiguousarray(out)


def fingerprint(raw):
    return raw.shape, raw.dtype.str, hashlib.sha256(np.ascontiguousarray(raw).tobytes()).digest()


def write_result(siril, data, meta, expected, label):
    """Apply to Siril with an Undo state, only if the image is still the one that was processed."""
    final = encode_pixels(data, meta)
    with siril.image_lock():
        current = read_pixels(siril)
        if fingerprint(current) != expected:
            raise RuntimeError('The image in Siril changed since Apply started. Nothing was written.')
        if siril.undo_save_state(f'{APP_NAME} {label}') is False:
            raise RuntimeError('Siril could not save Undo. Pixels were not overwritten.')
        if siril.set_image_pixeldata(final) is False:
            raise RuntimeError('Siril refused the modified pixels.')
        actual = read_pixels(siril)
        if actual.shape != final.shape or not np.allclose(actual, final, rtol=0, atol=0 if final.dtype == np.uint16 else 2e-7, equal_nan=True):
            raise RuntimeError('Siril readback differs from the result. Use Siril Undo if needed.')
    return fingerprint(actual)


# ----------------------------------------------------------------------------------------------- UI
STYLE = """
QWidget { background: #242424; color: #dddddd; font-size: 10pt; }
QWidget#Card { background: #303030; border-radius: 10px; }
QLabel { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: bold; color: #dddddd; }
QLabel#Note { color: #999999; font-size: 9pt; }
QLabel#Value { color: #dddddd; font-weight: bold; min-width: 40px; }
QPushButton { background: #505050; border: 1px solid #666666; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #616161; }
QPushButton:disabled { color: #777777; border-color: #444444; }
QPushButton#Apply { background: #1274d5; color: #ffffff; font-weight: bold; border-color: #338be7; }
QPushButton#Apply:disabled { background: #354656; color: #808890; border-color: #43505c; }
QSlider::groove:horizontal { height: 4px; background: #505050; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1474d5; border-radius: 2px; }
QSlider::handle:horizontal { background: #dddddd; width: 14px; margin: -6px 0; border-radius: 7px; }
QProgressBar { border: 1px solid #444444; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #1474d5; }
"""


STRENGTH_TIP = 'How far to sharpen toward the sampling limit; always limited by the noise. 0 = Correct Only.'
STAR_TIP = 'Shrinks stars toward the sampling limit (noise-limited). Only this slider changes star size.'
HALO_TIP = 'Pulls star halos in; their light goes back into the star core (brightness kept).'
SHARP_TIP = ('Number of sharpening passes (PSF re-measured each time); 2.5 = 2 passes + a half-strength pass. 0 = Correct Only. '
             'Umbra Deconvolution 1.0: every step is sharper than the last; 8 is the strongest; cleaner dark dust lanes. '
             '1.1a experimental: sharper fine strands, less beading; sky mask on automatically; may keep slight coma tails on bright '
             'corner stars unless Protect star shapes is ticked. 1.0 stays the default. '
             'fb_a2: about 3-4; beading can start around 5. Images differ, so judge by eye.')
PROTECT_TIP = ('Fine-detail models only: stars and their coma tails come from Umbra Deconvolution 1.0 (best star shapes), all other '
               'detail from 1.1a. About 2x slower.')
OLD_MODEL_TIP = 'this model has no separate star controls; use a newer model'


def current_settings(strength_pct, correct_only=False, auto_psf=True, fwhm=3.0, star_pct=0, halo_pct=0):
    """UI state -> (strength 0..1, fwhm or None, star 0..1, halo 0..1). Correct Only forces strength = star = halo = 0."""
    s, st, h = (0.0, 0.0, 0.0) if correct_only else (strength_pct / 100.0, star_pct / 100.0, halo_pct / 100.0)
    return s, (None if auto_psf else float(fwhm)), st, h


def settings_label(s, fwhm, star=0.0, halo=0.0, sharpening=None):
    head = f'strength {s:.2f}' if sharpening is None else f'sharpening {float(sharpening):.2f}'
    return f'{head} stars {star:.2f} halo {halo:.2f} ' + ('auto PSF' if fwhm is None else f'PSF {fwhm:.1f} px')


def launch_gui(siril=None, fake_image=None, smoke=False):
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
                                 QProgressBar, QComboBox, QCheckBox)

    class Worker(QThread):
        done = pyqtSignal(object, float)
        failed = pyqtSignal(str)
        prog = pyqtSignal(int, int)

        def __init__(self, data, s, fwhm, star, halo, model_path=None, sharpening=None, drizzled=None, protect_stars=False):
            super().__init__()
            self.protect_stars = protect_stars
            self.data, self.s, self.fwhm, self.star, self.halo, self.model_path, self.sharpening = data, s, fwhm, star, halo, model_path, sharpening
            self.drizzled = drizzled

        def run(self):
            try:
                t0 = time.time()
                eng = get_engine(self.model_path)
                out = eng.run(self.data, self.s, self.fwhm, progress=lambda n, t: self.prog.emit(int(n), int(t)), star=self.star, halo=self.halo, sharpening=self.sharpening, drizzled=self.drizzled, protect_stars=self.protect_stars)
                self.done.emit(out, time.time() - t0)
            except Exception as e:
                self.failed.emit(f'{type(e).__name__}: {e}')

    class Task(QThread):
        done = pyqtSignal(object)
        failed = pyqtSignal(str)
        prog = pyqtSignal(int, int)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.done.emit(self.fn(lambda a, b: self.prog.emit(int(a), int(b))))
            except Exception as e:
                self.failed.emit(str(e) if isinstance(e, DownloadError) else f'{type(e).__name__}: {e}')

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f'{APP_NAME} {VERSION}')
            self.setStyleSheet(STYLE)
            self.worker = None
            self.meta = self.identity = None
            self.settings = None
            self.result = None
            self.setFixedWidth(380)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(8)
            lay.addWidget(QLabel(f'{APP_NAME} (preview)', objectName='Title'))
            lay.addWidget(QLabel('AI deconvolution for linear images.', objectName='Note', wordWrap=True))
            card = QWidget(objectName='Card')
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 12)
            mrow = QHBoxLayout()
            mrow.addWidget(QLabel('Model'))
            self.model_box = QComboBox()
            cur = default_model_path()
            for label, path in list_models():
                self.model_box.addItem(label, path)
            i = self.model_box.findData(cur) if cur else -1
            if i < 0 and cur:
                self.model_box.addItem(Path(cur).stem, cur)
                i = self.model_box.count() - 1
            if i >= 0:
                self.model_box.setCurrentIndex(i)
            mrow.addWidget(self.model_box, 1)
            cl.addLayout(mrow)
            brow = QHBoxLayout()
            self.dl_btn = QPushButton('Download model')
            self.dl_btn.setToolTip('Download a model from this project\'s GitHub releases (checked against a SHA-256 before it is installed).')
            self.dl_btn.clicked.connect(self.show_dl_menu)
            self.upd_btn = QPushButton('Check for updates')
            self.upd_btn.setToolTip('Looks for newer models and a newer script. Nothing is downloaded or replaced until you ask.')
            self.upd_btn.clicked.connect(lambda: self.fetch_info(manual=True))
            brow.addWidget(self.dl_btn); brow.addWidget(self.upd_btn)
            cl.addLayout(brow)
            self.upd_note = QLabel('', objectName='Note', wordWrap=True)
            self.upd_note.setOpenExternalLinks(True)
            self.upd_note.setTextFormat(Qt.TextFormat.RichText)
            self.upd_note.hide()
            cl.addWidget(self.upd_note)
            self.info = None          # last check_updates() result
            self.net = None           # running network task
            self.correct = QCheckBox('Correct Only')
            self.correct.setToolTip(STRENGTH_TIP)
            cl.addWidget(self.correct)
            self.strength, self.sval = self.make_slider(cl, 'Strength', 0, 100, 50, lambda v: f'{v}%', STRENGTH_TIP)
            self.sharp, self.shval = self.make_slider(cl, 'Sharpening', 0, 32, int(SHARP_DEFAULT * 4), lambda v: f'{v / 4:.2f}', SHARP_TIP)   # 0..8 in 0.25 steps
            self.fb = is_fb(self.model_box.currentData())
            self.protect = QCheckBox('Protect star shapes (slower)')
            self.protect.setToolTip(PROTECT_TIP)
            cl.addWidget(self.protect)
            self.auto = QCheckBox('Automatic PSF')
            self.auto.setChecked(True)
            cl.addWidget(self.auto)
            self.psf, self.pval = self.make_slider(cl, 'PSF Diameter (px)', 15, 100, 30, lambda v: f'{v / 10:.1f}',
                                                   'Manual star diameter (FWHM) in pixels.')
            self.psf.setEnabled(False)
            self.sharpen, _ = self.make_slider(cl, 'Sharpen Stars', 0, 100, 0, lambda v: f'{v}%', STAR_TIP)
            self.halo, _ = self.make_slider(cl, 'Halo Reduction', 0, 100, 0, lambda v: f'{v}%', HALO_TIP)
            self.star_capable = False   # set from the loaded model's cfg.n_cond (>= 6)
            self.update_controls()
            self.correct.toggled.connect(lambda c: self.update_controls())
            self.auto.toggled.connect(lambda a: self.psf.setEnabled(not a))
            self.model_box.currentIndexChanged.connect(lambda *_: self.model_changed())
            lay.addWidget(card)
            self.apply_btn = QPushButton('Apply', objectName='Apply')
            self.apply_btn.clicked.connect(self.start)
            lay.addWidget(self.apply_btn)
            self.progress = QProgressBar()
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
            self.progress.hide()
            lay.addWidget(self.progress)
            self.idle_msg = ('Set the controls, then press Apply. The image in Siril is replaced (Undo available).'
                             if siril is not None else 'Test mode (no Siril).')
            self.note = QLabel(self.idle_msg, objectName='Note', wordWrap=True)
            lay.addWidget(self.note)
            close = QPushButton('Close')
            close.clicked.connect(self.close)
            lay.addWidget(close)

        def update_controls(self):
            co = self.correct.isChecked()
            self.strength.setEnabled(not co)
            self.sharp.setEnabled(not co)
            for w in (self.strength, self.strength.label_widget, self.sval):   # fb models: Sharpening replaces Strength
                w.setVisible(not self.fb)
            for w in (self.sharp, self.sharp.label_widget, self.shval):
                w.setVisible(self.fb)
            self.protect.setVisible(is_finedetail(self.model_path()))
            for sl, tip in ((self.sharpen, STAR_TIP), (self.halo, HALO_TIP)):
                sl.setEnabled(self.star_capable and not co)
                t = tip if self.star_capable else OLD_MODEL_TIP
                sl.setToolTip(t)
                sl.label_widget.setToolTip(t)

        def model_path(self):
            return self.model_box.currentData()

        # ---- model download / update check (a button press starts every network action, except the manifest lookup at launch)
        def run_net(self, fn, on_done, busy_text):
            if self.net is not None and self.net.isRunning():
                return
            self.net = Task(fn)
            self.net.prog.connect(self.net_progress)
            self.net.done.connect(lambda r: (self.net_idle(), on_done(r)))
            self.net.failed.connect(lambda m: (self.net_idle(), self.net_failed(m)))
            for b in (self.dl_btn, self.upd_btn, self.apply_btn):
                b.setEnabled(False)
            self.progress.setRange(0, 0); self.progress.show()
            self.set_status(busy_text)
            self.net.start()

        def net_progress(self, got, total):
            if total > 0:
                self.progress.setRange(0, 1000); self.progress.setValue(int(1000 * got / total))
                self.set_status(f'Downloading… {got / 1e6:.0f} / {total / 1e6:.0f} MB')

        def net_idle(self):
            self.progress.hide()
            self.dl_btn.setEnabled(True); self.upd_btn.setEnabled(True)
            self.apply_btn.setEnabled(bool(self.model_path()))

        def net_failed(self, msg):
            self.set_status(msg + ('' if self.model_path() else ' No model is installed yet.'), True)

        def refresh_models(self, select=None):
            cur = select or self.model_path() or default_model_path()
            self.model_box.blockSignals(True)
            self.model_box.clear()
            for label, path in list_models():
                self.model_box.addItem(label, path)
            i = self.model_box.findData(cur) if cur else -1
            self.model_box.setCurrentIndex(max(i, 0) if self.model_box.count() else -1)
            self.model_box.blockSignals(False)
            self.apply_btn.setEnabled(bool(self.model_path()))
            self.dl_btn.setProperty('attention', not self.model_path())
            self.dl_btn.setStyleSheet('' if self.model_path() else 'background: #1274d5; color: #ffffff; font-weight: bold;')
            self.model_changed(remember=bool(select))

        def fetch_info(self, manual=False, then=None):
            if offline_mode():
                self.upd_note.setText('Offline mode (UMBRA_DECONV_OFFLINE is set); using the installed models.'); self.upd_note.show()
                self.net_idle(); return
            def work(prog):
                man = fetch_manifest()
                return check_updates(man)
            def done(res):
                self.info = res
                if manual:
                    from html import escape
                    lines = describe_updates(res)
                    html = '<br>'.join(escape(l).replace(escape(res['script_url'] or ' '), f'<a href="{escape(res["script_url"])}">release page</a>') if res['script_url'] else escape(l) for l in lines)
                    self.upd_note.setText(html); self.upd_note.show()
                if then:
                    then()
                elif not self.model_path():
                    d = next((m for m, st in res['models'] if st == 'missing' and m.get('default')), None) or next((m for m, st in res['models'] if st == 'missing'), None)
                    if d:
                        self.set_status(f"No model installed. Press Download model to get {d['name']} ({d['size_bytes'] / 1e6:.0f} MB).")
            def failed(msg):
                self.info = None
                if manual or not self.model_path():
                    self.upd_note.setText(msg); self.upd_note.show()
                self.set_status(msg if not self.model_path() else self.idle_msg, not self.model_path())
            self.net = Task(work)
            self.net.done.connect(lambda r: (self.net_idle(), done(r)))
            self.net.failed.connect(lambda m: (self.net_idle(), failed(m)))
            for b in (self.dl_btn, self.upd_btn, self.apply_btn):
                b.setEnabled(False)
            self.set_status('Checking the model list…')
            self.net.start()

        def show_dl_menu(self):
            if self.info is None:
                self.fetch_info(manual=False, then=self.show_dl_menu if not offline_mode() else None)
                return
            from PyQt6.QtWidgets import QMenu
            menu = QMenu(self)
            for m, st in self.info['models']:
                if st == 'current':
                    continue
                tag = 'Update ' if st == 'update' else 'Download '
                menu.addAction(f"{tag}{m['name']} ({m['size_bytes'] / 1e6:.0f} MB)").triggered.connect(lambda _=False, e=m: self.start_download(e))
            if menu.isEmpty():
                self.set_status('All published models are already installed and up to date.')
                return
            menu.exec(self.dl_btn.mapToGlobal(self.dl_btn.rect().bottomLeft()))

        def start_download(self, entry):
            def work(prog):
                return download_model(entry, MODEL_DIR, progress=prog)
            def done(path):
                self.info = None
                self.refresh_models(select=str(path))
                self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
            self.run_net(work, done, f"Downloading {entry['name']}…")

        def model_changed(self, *_, remember=True):
            path = self.model_path()
            if not path:
                self.set_status(f'No model installed. Press Download model (or put a *.onnx file in {MODEL_DIR}).', True)
                return
            self.set_status('Loading model…')
            QApplication.processEvents()
            try:
                self.star_capable = get_engine(path).has_star_controls
                self.fb = is_fb(path)
                self.update_controls()
                if remember:
                    remember_model(path)
                self.set_status(self.idle_msg)
            except Exception as e:
                self.set_status(f'Could not load the model: {e}', True)

        def make_slider(self, layout, text, lo, hi, default, fmt, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            val = QLabel(fmt(default), objectName='Value')
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            layout.addLayout(row)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(default)
            s.valueChanged.connect(lambda v, l=val: l.setText(fmt(v)))
            if tip:
                label.setToolTip(tip)
                s.setToolTip(tip)
            layout.addWidget(s)
            s.label_widget = label
            return s, val

        def set_status(self, msg, error=False):
            self.note.setText(msg)
            self.note.setStyleSheet('color: #e05050;' if error else '')

        def on_progress(self, n, total):
            if total > 0:
                self.progress.setRange(0, total)
                self.progress.setValue(n)
                self.set_status(f'Running… {n} / {total} tiles (all passes)')

        def start(self):
            if self.worker is not None and self.worker.isRunning():
                return
            if not self.model_path():
                self.set_status('No model installed. Press Download model first.', True)
                return
            sp = self.sharpen.value() if self.star_capable else 0
            hp = self.halo.value() if self.star_capable else 0
            self.settings = current_settings(self.strength.value(), self.correct.isChecked(), self.auto.isChecked(), self.psf.value() / 10.0, sp, hp)
            try:
                if siril is not None:
                    with siril.image_lock():
                        raw = read_pixels(siril)
                        data, self.meta = decode_pixels(raw)
                        self.identity = fingerprint(raw)
                        self.drizzled = siril_drizzled(siril)
                else:
                    data, self.meta = fake_image, dict(dtype=np.dtype('float32'), shape=fake_image.shape)
                    self.drizzled = False
            except Exception as e:
                self.set_status(f'Could not read the image: {e}', True)
                return
            w = stretched_warning(data)
            if w:
                msg = f'{APP_NAME}: Warning: {w}'
                if siril is not None:
                    siril.log(msg)
                else:
                    print(msg, flush=True)
            self.apply_btn.setEnabled(False)
            self.progress.setRange(0, 0)
            self.progress.show()
            self.set_status('Running…' + (' (first run loads the model)' if str(self.model_path()) not in _ENGINE else ''))
            self.run_sharp = (0.0 if self.correct.isChecked() else self.sharp.value() / 4.0) if self.fb else None
            self.worker = Worker(data, *self.settings, model_path=self.model_path(), sharpening=self.run_sharp, drizzled=self.drizzled,
                                 protect_stars=self.protect.isChecked() and is_finedetail(self.model_path()))
            self.worker.prog.connect(self.on_progress)
            self.worker.done.connect(self.finished)
            self.worker.failed.connect(self.fail)
            self.worker.start()

        def finished(self, out, secs):
            label = settings_label(*self.settings, sharpening=self.run_sharp)
            fw = get_engine(self.worker.model_path).last_fwhm
            fw_txt = f' Measured FWHM {fw:.2f} px.' if fw is not None else ''
            note = get_engine(self.worker.model_path).last_note
            if note:
                fw_txt += ' ' + note
            try:
                if siril is not None:
                    self.identity = write_result(siril, out, self.meta, self.identity, label)
                    siril.log(f'{APP_NAME}: {label} done in {secs:.1f} s (Undo available).{fw_txt}')
                self.result = out
                self.progress.hide()
                self.apply_btn.setEnabled(True)
                self.set_status(f'Done in {secs:.1f} s.{fw_txt} ' + ('Applied to Siril; use Siril Undo to revert.' if siril is not None else ''))
            except Exception as e:
                self.fail(str(e))

        def fail(self, msg):
            self.progress.hide()
            self.apply_btn.setEnabled(True)
            self.set_status(msg, True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = Window()
    win.show()
    QTimer.singleShot(0, lambda: win.model_changed(remember=False))
    win.apply_btn.setEnabled(bool(win.model_path()))
    if not smoke:
        QTimer.singleShot(200, lambda: win.fetch_info(manual=False))      # manifest lookup only (no download, nothing sent but a generic client name)
    if smoke:
        def poll():
            if win.result is not None:
                ok = win.result.shape == fake_image.shape and bool(np.isfinite(win.result).all())
                print('UI TEST', 'OK' if ok else 'BAD OUTPUT', win.note.text(), flush=True)
                app.quit()
            elif win.worker is not None and win.worker.isFinished():
                print('UI TEST FAILED', win.note.text(), flush=True)
                app.quit()
            else:
                QTimer.singleShot(300, poll)
        win.model_changed(remember=False)   # load now so the control states reflect the model's n_cond
        assert current_settings(50) == (0.5, None, 0.0, 0.0) and current_settings(50, True, True, 3.0, 80, 60) == (0.0, None, 0.0, 0.0)
        assert current_settings(100, False, False, 3.5, 40, 20) == (1.0, 3.5, 0.4, 0.2)
        assert 'stars 0.40' in settings_label(1.0, 3.5, 0.4, 0.2) and 'halo 0.20' in settings_label(1.0, 3.5, 0.4, 0.2)
        assert is_fb('x/umbra_deconv_fb_a2.onnx') and not is_fb('x/umbra_deconv_b1.onnx') and pass_plan(2.5) == [1.0, 1.0, 0.5] and pass_plan(0) == [] and pass_plan(9) == [1.0] * 8
        assert win.sharp.isVisible() == win.fb and win.strength.isVisible() == (not win.fb)
        assert win.strength.value() == 50 and win.sharpen.value() == 0 and win.halo.value() == 0 and win.auto.isChecked() and not win.psf.isEnabled()
        capable = win.star_capable
        assert capable == get_engine(win.model_path()).has_star_controls
        assert win.sharpen.isEnabled() == capable and win.halo.isEnabled() == capable
        if not capable:
            assert win.sharpen.toolTip() == OLD_MODEL_TIP and win.halo.toolTip() == OLD_MODEL_TIP
        else:
            assert win.sharpen.toolTip() == STAR_TIP and win.halo.toolTip() == HALO_TIP
        win.correct.setChecked(True)
        assert not win.strength.isEnabled() and not win.sharpen.isEnabled() and not win.halo.isEnabled()
        win.correct.setChecked(False)
        assert win.strength.isEnabled() and win.sharpen.isEnabled() == capable and win.halo.isEnabled() == capable
        win.star_capable = True; win.update_controls()
        assert win.sharpen.isEnabled() and win.halo.isEnabled() and win.sharpen.toolTip() == STAR_TIP
        win.correct.setChecked(True)
        assert not win.sharpen.isEnabled() and not win.halo.isEnabled()
        win.correct.setChecked(False)
        win.star_capable = capable; win.update_controls()
        QTimer.singleShot(300, win.start)
        QTimer.singleShot(1000, poll)
    return app.exec()


# ----------------------------------------------------------------------------------------------- entry
def run_file(src, dst, s, fwhm, model_path=None, star=0.0, halo=0.0, sharpening=None, protect_stars=False):
    from astropy.io import fits
    src, dst = Path(src).resolve(), Path(dst).resolve()
    explicit_same = dst == src
    if explicit_same:
        raise SystemExit('Refusing to overwrite the input file (--out equals --image).')
    if dst.exists():
        raise SystemExit(f'Output exists, refusing to overwrite: {dst}')
    with fits.open(src, memmap=False) as h:
        hdu = next(x for x in h if x.data is not None)
        data = np.asarray(hdu.data, np.float32)
        header = hdu.header.copy()
        drizzled = is_drizzled(' '.join(repr(x.header) for x in h) + ' ' + ' '.join(str(x.name) for x in h) + ' ' + src.name)
    w = stretched_warning(data)
    if w:
        print(f'{APP_NAME}: Warning: {w}', flush=True)
    t0 = time.time()
    eng = get_engine(model_path)
    if is_fb(eng.model_path) and sharpening is None: sharpening = SHARP_DEFAULT
    if not is_fb(eng.model_path): sharpening = None                       # older models: single Strength pass (0.0.6 behaviour)
    out = eng.run(data, s, fwhm, star=star, halo=halo, sharpening=sharpening, drizzled=drizzled, protect_stars=protect_stars)
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        header.pop(k, None)
    header['HISTORY'] = f'{APP_NAME} {VERSION} {settings_label(s, fwhm, star, halo, sharpening)}' + (' protect-stars' if protect_stars else '')
    fits.PrimaryHDU(out.astype(np.float32), header=header).writeto(dst)
    print(f'wrote {dst} shape={out.shape} finite={bool(np.isfinite(out).all())} device={eng.device} model={Path(eng.model_path).name} '
          f'measured_fwhm={eng.last_fwhm} {time.time() - t0:.1f} s' + (f' NOTE: {eng.last_note}' if eng.last_note else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='standalone: FITS file to process (never modified)')
    ap.add_argument('--out', help='standalone: output FITS (must not exist)')
    ap.add_argument('--strength', type=float, default=0.5, help='0..1 (default 0.5)')
    ap.add_argument('--stars', type=float, default=0.0, help='Sharpen Stars 0..1 (default 0; needs an n_cond>=6 model, ignored otherwise)')
    ap.add_argument('--halo', type=float, default=0.0, help='Halo Reduction 0..1 (default 0; needs an n_cond>=6 model, ignored otherwise)')
    ap.add_argument('--correct-only', action='store_true', help='force strength, stars and halo to 0')
    ap.add_argument('--fwhm', type=float, help='manual PSF FWHM in px (default: automatic)')
    ap.add_argument('--model', help='model file (*.onnx); default: the remembered / first model')
    ap.add_argument('--sharpening', type=float, help='fb models: passes 0..8 in steps of 0.25 (default 3; 2.5 = 2 passes + half pass); ignored for older models')
    ap.add_argument('--protect-stars', action='store_true', help='fine-detail models: star shapes from Umbra Deconvolution 1.0 (~2x slower)')
    ap.add_argument('--check-updates', action='store_true', help='compare installed models and this script with the published manifest, then exit')
    ap.add_argument('--download', nargs='?', const='default', metavar='MODEL_ID|all', help='download a published model (default: the default model) into the models folder, then exit')
    ap.add_argument('--ui-test', action='store_true', help='open the window on a fake image, Apply once, exit')
    a = ap.parse_args()
    if a.check_updates or a.download:
        try:
            man = fetch_manifest()
            res = check_updates(man)
            if a.check_updates:
                print('\n'.join(describe_updates(res)))
            if a.download:
                want = [e for e, st in res['models'] if st != 'current' and (a.download == 'all' or (a.download == 'default' and e.get('default')) or a.download == e['id'])]
                if not want:
                    print('Nothing to download (installed and up to date, or unknown model id).')
                for e in want:
                    last = [0]
                    def pr(g, t):
                        if g - last[0] >= 20_000_000 or g == t:
                            last[0] = g; print(f"  {e['name']}: {g / 1e6:.0f} / {t / 1e6:.0f} MB", flush=True)
                    print('installed', download_model(e, MODEL_DIR, pr))
            return 0
        except DownloadError as e:
            print(f'ERROR: {e}', file=sys.stderr)
            return 1
    if a.ui_test:
        rng = np.random.default_rng(0)
        fake = (0.02 + 0.003 * rng.standard_normal((3, 128, 128))).astype(np.float32)
        yy, xx = np.mgrid[:128, :128]
        for cy, cx in ((30, 40), (80, 90), (100, 25)):
            fake += (0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2)))[None].astype(np.float32)
        return launch_gui(None, fake_image=fake, smoke=True)
    if a.image:
        if not a.out:
            ap.error('--out is required with --image')
        if not 0 <= a.strength <= 1:
            ap.error('--strength must lie in [0, 1]')
        if a.fwhm is not None and not 0.5 <= a.fwhm <= 20:
            ap.error('--fwhm must be between 0.5 and 20 px')
        if a.sharpening is not None and not 0 <= a.sharpening <= 8:
            ap.error('--sharpening must lie in 0..8')
        if not 0 <= a.stars <= 1 or not 0 <= a.halo <= 1:
            ap.error('--stars and --halo must lie in [0, 1]')
        s, st, h = (0.0, 0.0, 0.0) if a.correct_only else (a.strength, a.stars, a.halo)
        run_file(a.image, a.out, s, a.fwhm, model_path=a.model, star=st, halo=h, sharpening=0.0 if a.correct_only else a.sharpening, protect_stars=a.protect_stars)
        return 0
    try:
        _import_engine()   # import onnxruntime before Qt starts; any failure is kept and shown in the window
    except Exception:
        pass
    import sirilpy as s
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd('requires', '1.4.0')
        return launch_gui(siril)
    finally:
        siril.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
