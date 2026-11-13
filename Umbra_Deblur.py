#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Umbra Deblur (BETA), Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
"""AI deblur (deconvolution) for linear images. Apply shows the result in the window; Apply to Siril replaces the image (Undo available).
Aberration Removal Only fixes star shapes (aberrations) without sharpening. Sharpening sets the number of passes (0-8).
Sharpen Stars is the only control that shrinks stars; Halo Reduction is the only one that pulls in star halos.
Automatic PSF measures the star size from the image; switch it off to set the PSF diameter by hand.
Run in Siril (Scripts menu), or standalone:
  python Umbra_Deblur.py --image in.fits --out out.fits [--model PATH.onnx] [--sharpening 0..8] [--strength 0..1] [--stars 0..1] [--halo 0..1] [--correct-only] [--fwhm PX] [--protect-stars]
  python Umbra_Deblur.py --ui-test        (opens the window on a fake image, runs Apply once, exits)
  python Umbra_Deblur.py --check-updates | --download [MODEL_ID|all]"""
VERSION = '0.3.0'
# 0.3.0: new window like Umbra Guided GHS: fixed controls column, zoomable image on the right that shows the original at once; Apply shows the result there, Apply to Siril writes it, Undo all puts back the starting image; Download sits next to the model list, models not installed are listed as (Not Downloaded), and the update check runs at every start (green / yellow note); Correct Only is now Aberration Removal Only; Automatic PSF is greyed out when the image has no stars.
# 0.2.0: GPU by default also on Linux with NVIDIA (CUDA libraries installed on first run if missing); OpenVINO / MIGraphX used when installed.
# 0.1.1: buttons, check boxes, drop-down menus and scroll bars now look the same as in Umbra Astro Studio.
# 0.1.0: Umbra Deblur 0.1.0 replaces Umbra Deconvolve 0.0.14: new name, models "Umbra Deblur v1.0" / "v1.1 (experimental)" (same files as before; the old
# umbra_deconvolution_* files and the old "Umbra Deconvolve" folder keep working), a Preview window, Cancel, the family window layout, remembered settings.
# ============================================================================================ shared code (src/common.py)
# Everything the five scripts share: dependency install, ONNX Runtime sessions, updater, preferences, Siril I/O, family UI, CLI.
import sys, os, json, hashlib, argparse, time, math, re, types
from pathlib import Path
_STANDALONE = any(a in ('--image', '--ui-test', '--download', '--check-updates') or a.startswith(('--image=', '--download=')) for a in sys.argv)
if __name__ == '__main__' and not _STANDALONE and '--help' not in sys.argv and '-h' not in sys.argv:
    import sirilpy as _siril_deps
    def _ort_requirement():
        """onnxruntime variant for this machine (all variants import as 'onnxruntime'; only one may be installed). An existing one is kept."""
        import importlib.metadata as _md, platform as _pf, shutil as _sh
        for n in ('onnxruntime-gpu', 'onnxruntime-directml', 'onnxruntime-rocm', 'onnxruntime-migraphx', 'onnxruntime-openvino', 'onnxruntime'):
            try:
                _md.version(n); return n
            except Exception:
                pass
        if _pf.system() == 'Windows': return 'onnxruntime-directml'          # GPU through DirectX 12 (NVIDIA / AMD / Intel), no extra downloads
        if _pf.system() == 'Linux' and _sh.which('nvidia-smi'): return 'onnxruntime-gpu'   # CUDA; the CUDA 12 / cuDNN 9 libraries are installed below if missing
        return 'onnxruntime'                                                   # macOS: Core ML is built in; Linux without NVIDIA: CPU
    _ORT_PKG = _ort_requirement()
    _siril_deps.ensure_installed('numpy', _ORT_PKG, 'PyQt6', 'scipy', 'astropy')
    if _ORT_PKG == 'onnxruntime-gpu':
        # GPU by default on NVIDIA: if the system has no CUDA 12 / cuDNN 9 libraries, install NVIDIA's pip packages of them
        # (onnxruntime loads them with preload_dlls()).  Large one-time download (about 2-3 GB); CPU fallback if anything fails.
        import ctypes.util as _cu
        if not _cu.find_library('cudnn'):
            try:
                _siril_deps.ensure_installed('nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'nvidia-cufft-cu12',
                                             'nvidia-curand-cu12', 'nvidia-cuda-nvrtc-cu12')
            except Exception as _e:
                print(f'Could not install the CUDA libraries ({_e}); the script will use the CPU.', flush=True)
import numpy as np
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINES = {}      # tool key -> engine namespace (filled by the build)
TOOLS = {}        # tool key -> ToolSpec (filled by the build)
USER_AGENT = 'UmbraAstro'    # generic client name only; nothing about the user or the machine is sent
_REDIRECT_HOSTS = {'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com', 'raw.githubusercontent.com'}
GITHUB_USER = 'Kishanhpatel'


class ToolSpec:
    """Everything that differs between the tools (the shared code reads only this)."""
    def __init__(self, **kw):
        self.legacy_repos = ()
        self.fallback_folder = None
        self.labels = {}
        self.default_stems = ()
        self.notes = []
        self.linear_only = True
        self.preview_whole = True        # Apply runs on the whole image and shows the result in the pane
        self.reuse_preview = True        # Apply to Siril with unchanged settings reuses the result shown
        self.alt = None                  # (button text, info key) second preview layer
        self.progress_unit = 'tile'
        self.extras = {}                 # extra model files: kind -> file stem prefix (e.g. {'critic': 'umbra_starless_critic'}); not shown in the model picker
        self.has_editor = False          # Starless: an Edit marks button opens the correction-marks editor
        self.inline_preview = True       # every tool: Apply shows the result in the pane, Apply to Siril writes it
        self.__dict__.update(kw)

    @property
    def title(self):
        return f'{self.name} (BETA)'

    # hooks (overridden by the tool modules)
    def build_panel(self, ui, layout, host):
        raise NotImplementedError
    def add_cli(self, ap):
        pass
    def params_from_args(self, a):
        return {}
    def image_params(self, header_text):
        return {}
    def preview_params(self, params):
        return params
    def label(self, params):
        return ''
    def output(self, params, out, info):
        return out
    def post_apply(self, ctx, params, info):
        return ''
    def check_cli(self, a, ap):
        pass
    def post_cli(self, a, dst, info, header, label):
        return ''
    def prepare_job(self, view, job):
        """GUI thread, before a run starts: whatever the run needs from the window (snapshot).  Handed to engine_run as `extra`."""
        return None
    def engine_run(self, eng, data, params, progress, cancel, extra):
        return eng.run(data, params, progress=progress, cancel=cancel)


def env(spec, name, default=None):
    return os.environ.get(spec.prefix + name, default)


def offline_mode(spec):
    return bool(env(spec, 'OFFLINE') or os.environ.get('UMBRA_OFFLINE'))


# ----------------------------------------------------------------------------------------------- models folder
def _onnx_files(d):
    try:
        return sorted(Path(d).glob('*.onnx')) if Path(d).is_dir() else []
    except OSError:
        return []


def model_dir(spec):
    """(folder, used_fallback). <PREFIX>MODEL_DIR wins; else '<folder>' next to the script; Deblur falls back to the old
    'Umbra Deconvolve' folder when its own is missing or empty and the old one has models."""
    e = env(spec, 'MODEL_DIR')
    if e:
        return Path(e), False
    main = SCRIPT_DIR / spec.folder
    if spec.fallback_folder and not _onnx_files(main) and _onnx_files(SCRIPT_DIR / spec.fallback_folder):
        return SCRIPT_DIR / spec.fallback_folder, True
    return main, False


def download_dir(spec):
    e = env(spec, 'MODEL_DIR')
    return Path(e) if e else SCRIPT_DIR / spec.folder


_MANIFEST_NAMES = {}      # (tool, file stem) -> name from the manifest (after an update check)


def model_label(spec, stem):
    return spec.labels.get(stem) or _MANIFEST_NAMES.get((spec.key, stem)) or stem


def is_extra(spec, name):
    """True for a file of an extra kind (e.g. the Auto-keep critic): it is a model file, but never offered as a model."""
    stem = Path(str(name)).stem
    return any(stem.startswith(pre) for pre in spec.extras.values())


def extra_path(spec, kind):
    """Newest installed file of an extra kind (e.g. 'critic') in the models folder, or None.  <PREFIX><KIND>_MODEL overrides."""
    e = env(spec, kind.upper() + '_MODEL')
    if e:
        return e if Path(e).is_file() else None
    pre = spec.extras.get(kind)
    if not pre:
        return None
    files = [f for f in _onnx_files(download_dir(spec)) if f.stem.startswith(pre)]
    if not files:
        files = [f for f in _onnx_files(model_dir(spec)[0]) if f.stem.startswith(pre)]
    return str(sorted(files, key=lambda f: (_version_tuple(f.stem.rsplit('_v', 1)[-1].replace('_experimental', '')) or (0,), f.name))[-1]) if files else None


def list_models(spec):
    items = [(model_label(spec, f.stem), str(f)) for f in _onnx_files(model_dir(spec)[0]) if not is_extra(spec, f)]
    items.sort(key=lambda it: (0 if Path(it[1]).stem in spec.default_stems else 1, it[0]))
    extra = getattr(spec, 'extra_models', None)              # e.g. Starless: StarNet++ when the user has it installed
    if callable(extra):
        try:
            items += list(extra())
        except Exception:
            pass
    return items


def default_model_path(spec, saved=None):
    e = env(spec, 'MODEL')
    if e:
        return e
    d = model_dir(spec)[0]
    if saved:
        p = d / Path(saved).name
        if p.suffix == '.onnx' and p.is_file():
            return str(p)
    items = list_models(spec)
    return items[0][1] if items else None


# ----------------------------------------------------------------------------------------------- preferences
class Prefs:
    """Umbra_<Tool>_prefs.json: {"model": <file name>, "settings": {...}}; combined: {"tool": key, "tools": {key: {...}}}."""
    def __init__(self, path, combined=False, enabled=True):
        self.path, self.combined, self.enabled = Path(path), combined, enabled

    def _read(self):
        try:
            d = json.loads(self.path.read_text())
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def load(self, key):
        d = self._read()
        if self.combined:
            d = (d.get('tools') or {}).get(key) or {}
        s = d.get('settings')
        return (d.get('model') if isinstance(d.get('model'), str) else None), (s if isinstance(s, dict) else {})

    def load_tool(self):
        t = self._read().get('tool')
        return t if isinstance(t, str) else None

    def save(self, key, model, settings, current=None):
        if not self.enabled:
            return
        try:
            entry = {'model': Path(model).name if model else None, 'settings': settings}
            if self.combined:
                d = self._read()
                tools = d.get('tools') if isinstance(d.get('tools'), dict) else {}
                tools[key] = entry
                out = {'tool': current or key, 'tools': tools}
            else:
                out = entry
            self.path.write_text(json.dumps(out, indent=1))
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------- ONNX Runtime
_IMPORT_ERROR = {}   # first onnxruntime / scipy import failure (remembered so every later attempt shows the real cause)


def log_file(spec):
    return SCRIPT_DIR / f'{spec.script}_error.log'


def write_error_log(spec, text):
    try:
        log_file(spec).write_text(f'{time.ctime()}\npython {sys.executable}\n\n{text}')
    except Exception:
        pass


def import_engine(spec):
    """Import onnxruntime and scipy exactly once; on failure keep the ORIGINAL error (with traceback) for every later attempt."""
    if _IMPORT_ERROR:
        raise RuntimeError(_IMPORT_ERROR['msg'])
    try:   # Siril's working folder may have been deleted; some imports then fail in os.getcwd() -> move to a folder that exists
        os.getcwd()
    except OSError:
        os.chdir(str(SCRIPT_DIR))
    try:
        import onnxruntime  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        _IMPORT_ERROR['msg'] = f'{type(e).__name__}: {e} (full details: {log_file(spec)})'
        write_error_log(spec, f'sys.path {sys.path}\n\n{tb}')
        print(tb, flush=True)
        raise RuntimeError(_IMPORT_ERROR['msg']) from e


def make_session_factory(prefix, skip=()):
    """-> factory(model_path, feed) -> (run_fn, provider_label, note).  run_fn(feed_dict) -> list of outputs (like sess.run(None, feed)).
    GPU providers are tried in order of preference for the platform (CUDA, DirectML, ROCm, MIGraphX, OpenVINO, Core ML); each is checked against the CPU
    provider on one probe feed (outputs must agree to 1e-3 relative) and the first that works is used. Anything missing, failing or
    disagreeing falls through to the next, and finally to the CPU.  <PREFIX>CPU=1 forces the CPU."""
    def factory(model_path, feed, dims=None):
        import onnxruntime as ort
        path = str(model_path)
        def new_opts():
            o = ort.SessionOptions(); o.log_severity_level = 3
            for k, v in (dims or {}).items():          # pin free dimensions (Core ML compiles a fixed shape fast, a free one very slowly)
                o.add_free_dimension_override_by_name(k, int(v))
            return o
        so = new_opts()
        holder = []
        def mk_cpu():
            if not holder:
                holder.append(ort.InferenceSession(path, so, providers=['CPUExecutionProvider']))
            return holder[0]
        note = None
        sess, label = None, 'CPU'
        if not os.environ.get(prefix + 'CPU'):
            avail = ort.get_available_providers()
            if 'CUDAExecutionProvider' in avail and hasattr(ort, 'preload_dlls'):
                try:
                    ort.preload_dlls()          # CUDA / cuDNN from NVIDIA's pip packages, when the system has none
                except Exception:
                    pass
            cands = [('CUDAExecutionProvider', 'CUDA', {'use_tf32': '0', 'cudnn_conv_algo_search': 'HEURISTIC'}, False),
                     ('DmlExecutionProvider', 'DirectML', {}, True),
                     ('ROCMExecutionProvider', 'ROCm', {}, False),
                     ('MIGraphXExecutionProvider', 'MIGraphX', {}, False),
                     ('OpenVINOExecutionProvider', 'OpenVINO', {'device_type': 'GPU'}, False),
                     ('CoreMLExecutionProvider', 'Core ML', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}, False)]
            notes = []
            for name, lab, opts, dml in cands:
                if name not in avail or lab in skip:
                    continue
                try:
                    so2 = new_opts()
                    if dml:                                    # DirectML requirements
                        so2.enable_mem_pattern = False; so2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    g = ort.InferenceSession(path, so2, providers=[(name, opts), 'CPUExecutionProvider'])
                    if name not in g.get_providers():
                        raise RuntimeError('provider did not load')
                    a, b = g.run(None, feed), mk_cpu().run(None, feed)
                    err = max(float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-9)) for x, y in zip(a, b))
                    if np.isfinite(err) and err < 1e-3:
                        sess, label = g, lab
                        break
                    notes.append(f'{lab} result differed from the CPU ({err:.1e})')
                except Exception as e:
                    notes.append(f'{lab} unavailable ({str(e).strip().splitlines()[0][:90] if str(e).strip() else type(e).__name__})')
            if sess is None and notes:
                note = '; '.join(notes) + '; using the CPU.'
            elif sess is None:
                note = 'No GPU provider in this onnxruntime build; using the CPU.'
        if sess is None:
            sess = mk_cpu()
        return (lambda f: sess.run(None, f)), label, note
    return factory


_ENGINE_CACHE = {}     # tool key -> (model path, Engine)   (one model in memory per tool; the combined window unloads the others)


def get_engine(spec, model_path):
    c = _ENGINE_CACHE.get(spec.key)
    if c and c[0] == str(model_path):
        return c[1]
    _ENGINE_CACHE.pop(spec.key, None)
    import_engine(spec)
    p = Path(str(model_path or ''))
    if not p.is_file():
        raise FileNotFoundError(f'Model not found: {p}')
    eng = ENGINES[spec.key].Engine(str(p), make_session_factory(spec.prefix))
    if callable(getattr(spec, 'configure_engine', None)):
        spec.configure_engine(eng)
    _ENGINE_CACHE[spec.key] = (str(p), eng)
    return eng


def unload_engines(keep=None):
    for k in list(_ENGINE_CACHE):
        if k != keep:
            _ENGINE_CACHE.pop(k, None)
    import gc; gc.collect()


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


# ----------------------------------------------------------------------------------------------- model download / updates
class DownloadError(Exception):
    pass


class UpdSpec:
    """Where a script looks for models: its manifest repo and every repo whose release assets / release page it accepts."""
    def __init__(self, repo, allowed=None):
        self.repo = repo
        self.allowed = list(allowed or [repo])
        self.manifest_url = f'https://raw.githubusercontent.com/{repo}/main/models/models.json'
        self.releases_url = f'https://github.com/{repo}/releases'


def updspec_for(spec):
    return UpdSpec(spec.repo, [spec.repo, *spec.legacy_repos])


def combined_upd(specs):
    """The combined script's own manifest repo; it accepts model / script URLs from the repos of all its tools (no re-upload)."""
    me = f'{GITHUB_USER}/umbra-astro-ai'
    return UpdSpec(me, [me] + [t.repo for t in specs] + [r for t in specs for r in t.legacy_repos])


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


def _valid_model_url(url, file, upd):
    from urllib.parse import urlparse
    u = urlparse(str(url))
    return (u.scheme == 'https' and u.hostname == 'github.com' and not u.query and u.path.endswith('/' + file)
            and any(u.path.startswith(f'/{r}/releases/download/') for r in upd.allowed))


def _valid_entry(m, upd):
    try:
        f = m['file']
        return (re.fullmatch(r'[A-Za-z0-9._-]{1,120}\.onnx', f) is not None and '..' not in f and _valid_model_url(m['url'], f, upd)
                and re.fullmatch(r'[0-9a-f]{64}', str(m['sha256']).lower()) is not None and int(m['size_bytes']) > 0 and bool(m.get('name'))
                and re.fullmatch(r'[a-z]{1,20}', str(m.get('kind', 'model'))) is not None)
    except Exception:
        return False


def _version_tuple(v):
    try:
        return tuple(int(p) for p in str(v).strip().lstrip('v').split('.'))
    except Exception:
        return None


def parse_manifest(data, upd):
    """Validated manifest dict {'models': [...valid entries...], 'script': {'version','url'} or None}. Bad entries and entries marked
    "legacy" (kept only for old script versions) are dropped."""
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('models'), list):
        raise DownloadError('The model list has an unexpected format.')
    models = [dict(m, sha256=str(m['sha256']).lower(), size_bytes=int(m['size_bytes'])) for m in data['models']
              if isinstance(m, dict) and not m.get('legacy') and _valid_entry(m, upd)]
    sc = data.get('script')
    if not (isinstance(sc, dict) and _version_tuple(sc.get('version')) and any(str(sc.get('url', '')).startswith(f'https://github.com/{r}/') for r in upd.allowed)):
        sc = None
    return dict(models=models, script=sc)


_MANIFEST_CACHE = {}


def fetch_manifest(upd, url=None, use_cache=False):
    key = url or upd.manifest_url
    c = _MANIFEST_CACHE.get(key)
    if use_cache and c and time.time() - c[0] < 300:
        return c[1]
    try:
        with _open(key, 20) as resp:
            raw = resp.read(1_000_000)
        man = parse_manifest(json.loads(raw.decode('utf-8')), upd)
        _MANIFEST_CACHE[key] = (time.time(), man)
        return man
    except DownloadError:
        raise
    except Exception as e:
        why = ' Certificate problem: on macOS run "Install Certificates.command" for your Python.' if 'CERTIFICATE' in str(e).upper() else ''
        raise DownloadError(f'Could not reach the model list ({type(e).__name__}). Offline? The installed models still work.{why}') from e


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state(entry, folder):
    """'missing' | 'current' | 'update' for a manifest entry against the installed file."""
    p = Path(folder) / entry['file']
    if not p.is_file():
        return 'missing'
    # alt_sha256: older files that give identical results (e.g. the same weights stored at full size), so they are not re-downloaded
    ok = {entry['sha256']} | {str(h).lower() for h in (entry.get('alt_sha256') or []) if isinstance(h, str)}
    return 'current' if file_sha256(p) in ok else 'update'


def entries_for(manifest, tool=None):
    """Manifest entries for one tool (entries with a 'tool' field belong to that tool; without one they belong to every single script)."""
    return [m for m in manifest['models'] if tool is None or m.get('tool') in (None, tool)]


def check_updates(manifest, spec, folder=None):
    """-> dict(models=[(entry, state)], script_version, script_url, script_newer)."""
    folder = folder or download_dir(spec)
    sc = manifest.get('script')
    newer = bool(sc) and _version_tuple(sc['version']) > (_version_tuple(VERSION) or (0,))
    for m in manifest['models']:
        _MANIFEST_NAMES[(m.get('tool') or spec.key, Path(m['file']).stem)] = m['name']
    return dict(models=[(m, model_state(m, folder)) for m in entries_for(manifest, spec.key)],
                script_version=sc['version'] if sc else None, script_url=sc['url'] if sc else None, script_newer=newer)


def download_model(entry, upd, folder, progress=None, cancel=None):
    """Download one manifest entry into the models folder: temp file, size and SHA-256 check, then moved into place.  Returns the path."""
    if not _valid_entry(entry, upd):
        raise DownloadError('Refused: not a valid model entry.')
    d = Path(folder)
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


def describe_updates(res, spec):
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


def missing_model_message(spec, res=None):
    """Status text when no model is installed."""
    if res:
        d = next((m for m, st in res['models'] if st == 'missing' and m.get('default') and not m.get('kind')), None) or next((m for m, st in res['models'] if st == 'missing' and not m.get('kind')), None)
        if d:
            return f"No model installed. Select {d['name']} in the dropdown and click Download ({d['size_bytes'] / 1e6:.0f} MB)."
    return f'No model installed. Select a model in the dropdown and click Download, or put a .onnx file in {model_dir(spec)[0]}.'


# ----------------------------------------------------------------------------------------------- Siril I/O
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
        if siril.undo_save_state(label) is False:
            raise RuntimeError('Siril could not save Undo. Pixels were not overwritten.')
        if siril.set_image_pixeldata(final) is False:
            raise RuntimeError('Siril refused the modified pixels.')
        actual = read_pixels(siril)
        if actual.shape != final.shape or not np.allclose(actual, final, rtol=0, atol=0 if final.dtype == np.uint16 else 2e-7, equal_nan=True):
            raise RuntimeError('Siril readback differs from the result. Use Siril Undo if needed.')
    return fingerprint(actual)


def siril_header_text(siril):
    parts = []
    for fn in (lambda: siril.get_image_fits_header(return_as='str'), lambda: ' '.join(str(h) for h in (siril.get_image_history() or [])),
               lambda: siril.get_image_filename()):
        try:
            parts.append(str(fn()))
        except Exception:
            pass
    return ' '.join(parts)


def unique_path(p):
    p = Path(p)
    if not p.exists():
        return p
    for i in range(1, 1000):
        q = p.with_name(f'{p.stem}_{i}{p.suffix}')
        if not q.exists():
            return q
    raise FileExistsError(p)


def prefixed_path(siril, prefix):
    """<prefix><image file name> next to the image Siril has loaded, e.g. starmask_M42.fits (FITS extension kept, others -> .fit;
    never overwrites: _1, _2 ... when it exists).  Falls back to Siril's working folder."""
    name = None
    try:
        name = siril.get_image_filename()
    except Exception:
        pass
    if name and Path(name).parent.is_dir() and str(Path(name).parent) != '.':
        n = Path(name)
        stem = n.name[:-3] if n.name.lower().endswith('.gz') else n.name
        ext = Path(stem).suffix if Path(stem).suffix.lower() in ('.fit', '.fits', '.fts') else '.fit'
        return unique_path(n.with_name(prefix + Path(stem).stem + ext))
    wd = None
    try:
        wd = siril.get_siril_wd()
    except Exception:
        pass
    return unique_path(Path(wd or Path.home()) / f'{prefix}image.fit')


def side_path(siril, suffix):
    """<image>_<suffix>.fit next to the image Siril has loaded (falls back to Siril's working folder)."""
    name = None
    try:
        name = siril.get_image_filename()
    except Exception:
        pass
    if name and Path(name).parent.is_dir() and str(Path(name).parent) != '.':
        return unique_path(Path(name).with_name(Path(name).stem.removesuffix('.fit') + f'_{suffix}.fit'))
    wd = None
    try:
        wd = siril.get_siril_wd()
    except Exception:
        pass
    return unique_path(Path(wd or Path.home()) / f'umbra_{suffix}.fit')


def write_fits(path, data, header=None, history=''):
    from astropy.io import fits
    hdr = header.copy() if header is not None else fits.Header()
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        hdr.pop(k, None)
    if history:
        hdr['HISTORY'] = history
    fits.PrimaryHDU(np.asarray(data, np.float32), header=hdr).writeto(path)


def siril_fits_header(siril):
    try:
        from astropy.io import fits
        return fits.Header.fromstring(siril.get_image_fits_header(return_as='str'), sep='\n')
    except Exception:
        return None


def region_from_selection(siril, minimum=64):
    """Siril's current selection (x, y, w, h) when at least `minimum` px each way, else None."""
    try:
        sel = siril.get_siril_selection()
    except Exception:
        return None
    if sel and sel[2] >= minimum and sel[3] >= minimum:
        return tuple(int(v) for v in sel)
    return None


def crop_region(data, rect=None, size=768):
    """data (H,W) or (C,H,W) -> (cropped view copy, (x, y, w, h)).  rect given: that region; else a centred size x size crop (whole image if smaller)."""
    H, W = data.shape[-2:]
    if rect is None:
        w, h = min(size, W), min(size, H)
        rect = ((W - w) // 2, (H - h) // 2, w, h)
    x, y, w, h = rect
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = min(w, W - x), min(h, H - y)
    return np.ascontiguousarray(data[..., y:y + h, x:x + w]), (x, y, w, h)


# ----------------------------------------------------------------------------------------------- display (auto-stretch)
def _mtf(m, x):
    return (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)


def stretch_params(img, target=0.25, shadow_k=-2.8):
    """Linked auto-stretch (one set of parameters for all channels), like an STF: (c0, midtone)."""
    v = np.asarray(img, np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.5, 1.0
    if v.size > 2_000_000:
        v = v[::max(1, v.size // 2_000_000)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    c0 = min(max(0.0, med + shadow_k * mad), max(med - 1e-6, 0.0)) if mad > 0 else 0.0
    hi = float(np.percentile(v, 99.9)) if v.size else 1.0
    span = max(hi - c0, 1e-6) if hi > c0 else 1.0
    medn = min(max((med - c0) / span, 1e-6), 0.999)
    m = float(_mtf(target, medn))
    return c0, min(max(m, 1e-4), 0.9999), span


def apply_stretch(img, params):
    c0, m, span = params
    x = np.clip((np.nan_to_num(np.asarray(img, np.float32)) - c0) / span, 0, 1)
    y = _mtf(m, x)
    return np.clip(y, 0, 1)


def pool_for_display(img, max_side=1600):
    """Block-mean downsample (C,H,W) so the longer side is <= max_side."""
    H, W = img.shape[-2:]
    f = int(math.ceil(max(H, W) / max_side))
    if f <= 1:
        return img
    h, w = H // f * f, W // f * f
    a = np.nan_to_num(img[..., :h, :w]).reshape(*img.shape[:-2], h // f, f, w // f, f)
    return a.mean((-3, -1)).astype(np.float32)


def to_rgb8(img, params, flip=False):
    """(H,W) or (C,H,W) linear -> (H,W,3) uint8 with the given stretch."""
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = a[None]
    a = apply_stretch(a, params)
    if a.shape[0] == 1:
        a = np.repeat(a, 3, 0)
    rgb = np.ascontiguousarray((a.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8))
    return rgb[::-1].copy() if flip else rgb


# ----------------------------------------------------------------------------------------------- UI
STYLE = """
QWidget { background: #242424; color: #dddddd; font-size: 10pt; }
QWidget#Card { background: #303030; border-radius: 10px; }
QLabel, QCheckBox, QRadioButton, QSlider { background: transparent; }
QLabel#Title { font-size: 15pt; font-weight: bold; color: #dddddd; }
QLabel#Badge { color: #888888; font-size: 8pt; }
QLabel#Note { color: #999999; font-size: 9pt; }
QLabel#Value { color: #dddddd; font-weight: bold; min-width: 40px; }
QLabel#Caption { color: #aaaaaa; font-size: 9pt; }
QToolTip { color: #dddddd; background: #404040; border: 1px solid #666666; padding: 6px; }
QCheckBox { spacing: 4px; background: transparent; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #777777; border-radius: 3px; background: #3c3c3c; }
QCheckBox::indicator:checked { background: #1679db; border-color: #6eb5ff; }
QCheckBox::indicator:disabled { border-color: #555555; background: #2e2e2e; }
QComboBox { background: transparent; border: none; padding: 5px 25px 5px 9px; min-height: 18px; }
QComboBox QAbstractItemView { background: #3d3d3d; color: #dddddd; border: 1px solid #666666; border-radius: 9px; padding: 5px; selection-background-color: #1679db; outline: 0; }
QComboBox QAbstractItemView::item { padding: 5px 8px; border-radius: 5px; }
QPushButton { background: #505050; border: 1px solid #666666; border-radius: 5px; padding: 4px 7px; }
QPushButton:hover { background: #616161; }
QPushButton:checked { background: #1474d5; border-color: #358eef; }
QPushButton:disabled { color: #777777; border-color: #444444; }
QPushButton#Apply { background: #1274d5; color: #ffffff; font-weight: bold; border-color: #338be7; }
QPushButton#Apply:disabled { background: #354656; color: #808890; border-color: #43505c; }
QPushButton#ZoomButton { padding: 3px 10px; }
QScrollArea { border: none; background: #242424; }
QScrollBar:vertical { background: #242424; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #555555; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #242424; height: 8px; margin: 0; }
QScrollBar::handle:horizontal { background: #555555; border-radius: 4px; min-width: 24px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #505050; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #1474d5; border-radius: 2px; }
QSlider::handle:horizontal { background: #dddddd; width: 14px; margin: -6px 0; border-radius: 7px; }
QSlider::sub-page:horizontal:disabled { background: #555555; }
QProgressBar { border: 1px solid #444444; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #1474d5; }
"""

PANE_SIDE = 2400          # longest side of the images shown in the zoomable pane
IDLE_SIRIL = 'Set the controls, then press Preview or Apply. Apply replaces the image in Siril (Undo available).'
IDLE_TEST = 'Test mode (no Siril).'
IDLE_INLINE = 'Press Apply to see the result on the right, then Apply to Siril to write it (Undo available).'
_QT = {}


def _qt():
    """Build the Qt classes once (PyQt6 is imported only when a window is opened)."""
    if _QT:
        return _QT['ns']
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QProgressBar,
                                 QComboBox, QCheckBox, QSizePolicy, QStackedWidget, QMenu, QLayout)

    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QPainter, QPen, QColor
    from PyQt6.QtWidgets import QListView, QFrame, QGraphicsView, QGraphicsScene, QScrollArea
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QCursor, QPainterPath

    class ModernComboBox(QComboBox):
        """Rounded drop-down, drawn the same on every platform (same look as Umbra Astro Studio)."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumHeight(28)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            view = QListView(self); view.setFrameShape(QFrame.Shape.NoFrame); view.setSpacing(1); self.setView(view)

        def paintEvent(self, event):
            p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor('#4398ed') if self.hasFocus() else QColor('#666666'), 1))
            p.setBrush(QColor('#606060') if (self.underMouse() and self.isEnabled()) else QColor('#505050'))
            p.drawRoundedRect(QRectF(.5, .5, self.width() - 1, self.height() - 1), 8, 8)
            p.setPen(QColor('#dddddd') if self.isEnabled() else QColor('#888888')); p.setFont(self.font())
            text = self.fontMetrics().elidedText(self.currentText(), Qt.TextElideMode.ElideRight, max(1, self.width() - 35))
            p.drawText(self.rect().adjusted(9, 0, -27, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
            cx, cy = self.width() - 15, self.height() / 2
            p.setPen(QPen(QColor('#dddddd') if self.isEnabled() else QColor('#888888'), 1.7))
            p.drawLine(QPointF(cx - 4, cy - 2), QPointF(cx, cy + 2)); p.drawLine(QPointF(cx, cy + 2), QPointF(cx + 4, cy - 2)); p.end()

        def enterEvent(self, event):
            super().enterEvent(event); self.update()

        def leaveEvent(self, event):
            super().leaveEvent(event); self.update()

        def showPopup(self):
            super().showPopup()
            container = self.view().window()
            container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            container.setObjectName('RoundedComboPopup')
            container.setStyleSheet('QFrame#RoundedComboPopup {background:#3d3d3d;border:1px solid #666666;border-radius:10px;padding:4px;} '
                                    'QListView {background:#3d3d3d;color:#dddddd;border:none;border-radius:7px;padding:3px;outline:0;} '
                                    'QListView::item {padding:5px 8px;border-radius:5px;} QListView::item:selected {background:#1679db;}')

    class Worker(QThread):
        done = pyqtSignal(object, object, float, str, object)
        failed = pyqtSignal(str, bool)       # message, during model load
        cancelled = pyqtSignal()
        prog = pyqtSignal(int, int)
        stage = pyqtSignal(str)

        def __init__(self, spec, model_path, data, params, extra=None):
            super().__init__()
            self.spec, self.model_path, self.data, self.params, self.extra = spec, model_path, data, params, extra
            self.cancel_flag = False

        def run(self):
            loading = True
            try:
                t0 = time.time()
                if str(self.model_path) not in [c[0] for c in _ENGINE_CACHE.values()]:
                    self.stage.emit('Loading model…')
                eng = get_engine(self.spec, self.model_path)
                loading = False
                self.stage.emit('Running…')
                out, info = self.spec.engine_run(eng, self.data, self.params, lambda n, t: self.prog.emit(int(n), int(t)), lambda: self.cancel_flag, self.extra)
                self.done.emit(out, info, time.time() - t0, eng.provider, eng.note)
            except Exception as e:
                if not loading and str(e) == 'Cancelled':
                    self.cancelled.emit()
                    return
                import traceback
                tb = traceback.format_exc()
                write_error_log(self.spec, tb)
                self.failed.emit(f'{type(e).__name__}: {e}' if not loading else str(e), loading)

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

    def qimage(rgb):
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    class ZoomView(QGraphicsView):
        """The image area (like Umbra Guided GHS): fits the window until you zoom; wheel / pinch zooms around the mouse, drag pans,
        double-click fits again.  Swapping before / after keeps the zoom and position."""
        MAX_ZOOM = 8.0

        def __init__(self, pane):
            super().__init__()
            self.pane = pane
            self.setScene(QGraphicsScene(self))
            self.item = self.scene().addPixmap(QPixmap())
            self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setFrameShape(QFrame.Shape.NoFrame)
            self.setBackgroundBrush(QColor('#191919'))
            self.setStyleSheet('QGraphicsView { background: #191919; border: 1px solid #484848; border-radius: 7px; }')
            self.setMinimumSize(320, 240)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setToolTip('Scroll or pinch to zoom, drag to pan, double-click to fit. Hold Space (or Show original) for the original.')
            self.fitted = True
            self.placeholder = ''
            # brushes (Starless correction marks): the page that owns the strokes sets them up with set_brush()
            self.overlay = self.scene().addPixmap(QPixmap())
            self.overlay.setZValue(1)
            self.tool = 'navigate'                  # navigate | remove | keep | erase
            self.strokes = None                     # list owned by the tool page, or None = no brushes
            self.brush = dict(size=0.03, opacity=1.0, softness=0.35)
            self.colors = {'remove': (255, 60, 60), 'keep': (60, 220, 90), 'erase': (235, 235, 235)}
            self.on_stroke = None
            self.draft = None
            self.cursor_pos = None
            self.mpan = None
            self.max_strokes, self.max_points = 600, 4096
            self.setMouseTracking(True); self.viewport().setMouseTracking(True)

        # ---- brushes
        def set_brush(self, strokes=None, on_stroke=None):
            """strokes: the page's stroke list (None switches the brushes off and hides the marks)."""
            self.strokes, self.on_stroke = strokes, on_stroke
            if strokes is None:
                self.set_tool('navigate')
                self.overlay.setPixmap(QPixmap())

        def set_tool(self, tool):
            self.tool = tool if self.strokes is not None else 'navigate'
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if self.tool == 'navigate' else QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor if self.tool == 'navigate' else Qt.CursorShape.CrossCursor))
            self.viewport().update()

        def set_overlay(self, pm):
            self.overlay.setPixmap(pm if pm is not None else QPixmap())

        def image_shape(self):
            pm = self.item.pixmap()
            return (pm.height(), pm.width())

        def _norm(self, pos):
            h, w = self.image_shape()
            return [float(min(max((pos.x() - 0.5) / max(w - 1, 1), 0), 1)), float(min(max((pos.y() - 0.5) / max(h - 1, 1), 0), 1))]

        def _scene(self, p):
            h, w = self.image_shape()
            return QPointF(p[0] * max(w - 1, 1) + 0.5, p[1] * max(h - 1, 1) + 0.5)

        def _inside(self, pos):
            return not self.item.pixmap().isNull() and QRectF(self.item.pixmap().rect()).contains(pos)

        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.MiddleButton:
                self.mpan = e.position(); e.accept(); return
            if self.tool != 'navigate' and self.strokes is not None and e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                pos = self.mapToScene(e.position().toPoint())
                if self._inside(pos) and len(self.strokes) < self.max_strokes:
                    erase = self.tool == 'erase' or e.button() == Qt.MouseButton.RightButton
                    self.draft = dict(layer='both' if erase else self.tool, erase=bool(erase), size=float(self.brush['size']),
                                      softness=float(self.brush['softness']), opacity=float(self.brush['opacity']), points=[self._norm(pos)])
                    self.strokes.append(self.draft)
                    self.cursor_pos = pos
                    self.viewport().update()
                e.accept(); return
            super().mousePressEvent(e)

        def mouseMoveEvent(self, e):
            if self.mpan is not None:
                d = e.position() - self.mpan; self.mpan = e.position()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
                e.accept(); return
            pos = self.mapToScene(e.position().toPoint())
            self.cursor_pos = pos if self._inside(pos) else None
            if self.draft is not None:
                if self._inside(pos) and len(self.draft['points']) < self.max_points:
                    last = self._scene(self.draft['points'][-1])
                    if abs(pos.x() - last.x()) + abs(pos.y() - last.y()) >= 1.0:
                        self.draft['points'].append([round(v, 7) for v in self._norm(pos)])
                self.viewport().update(); e.accept(); return
            if self.tool != 'navigate':
                self.viewport().update(); e.accept(); return
            super().mouseMoveEvent(e)

        def mouseReleaseEvent(self, e):
            if e.button() == Qt.MouseButton.MiddleButton:
                self.mpan = None; e.accept(); return
            if self.draft is not None:
                self.draft = None
                self.viewport().update()
                if self.on_stroke is not None:
                    self.on_stroke()
                e.accept(); return
            super().mouseReleaseEvent(e)

        def leaveEvent(self, e):
            self.cursor_pos = None
            self.viewport().update()
            super().leaveEvent(e)

        def set_pixmap(self, pm, keep_view=True):
            same = (not self.item.pixmap().isNull()) and self.item.pixmap().size() == pm.size()
            self.item.setPixmap(pm)
            self.scene().setSceneRect(QRectF(pm.rect()))
            if not (keep_view and same) or self.fitted:
                self.fit()
            self.viewport().update()

        def clear(self, text=''):
            self.item.setPixmap(QPixmap())
            self.placeholder = text
            self.viewport().update()

        def fit(self):
            pm = self.item.pixmap()
            self.resetTransform()
            if not pm.isNull():
                self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
            self.fitted = True

        def zoom_by(self, factor):
            pm = self.item.pixmap()
            if pm.isNull():
                return
            vp = self.viewport()
            current = max(abs(self.transform().m11()), 1e-9)
            fitted = min(max(1, vp.width() - 4) / max(pm.width(), 1), max(1, vp.height() - 4) / max(pm.height(), 1))
            floor, ceiling = min(fitted, 1.0), max(self.MAX_ZOOM / max(self.devicePixelRatioF(), 1e-6), fitted)
            factor = min(max(factor, floor / current), ceiling / current)
            self.scale(factor, factor)
            self.fitted = abs(self.transform().m11() - fitted) < 1e-3 * fitted

        def wheelEvent(self, e):
            if self.draft is not None:
                e.accept(); return
            delta = e.angleDelta().y() or e.pixelDelta().y()
            if delta:
                self.zoom_by(1.2 if delta > 0 else 1 / 1.2)
            e.accept()

        def viewportEvent(self, e):
            if e.type() == QEvent.Type.NativeGesture and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(math.exp(float(e.value())))
                return True
            return super().viewportEvent(e)

        def mouseDoubleClickEvent(self, e):
            if self.tool == 'navigate':
                self.fit()
            e.accept()

        def resizeEvent(self, e):
            super().resizeEvent(e)
            if self.fitted:
                self.fit()

        def keyPressEvent(self, e):
            self.pane.keyPressEvent(e)

        def keyReleaseEvent(self, e):
            self.pane.keyReleaseEvent(e)

        def drawForeground(self, p, rect):
            if self.item.pixmap().isNull() and self.placeholder:
                p.save(); p.resetTransform()
                p.setPen(QColor('#9a9a9a'))
                p.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.placeholder)
                p.restore()
            h, w = self.image_shape()
            if self.draft is not None:          # live trail while the mouse is down (the marks image is rebuilt on release)
                pts = [self._scene(q) for q in self.draft['points']]
                pen = QPen(QColor(*self.colors['erase' if self.draft['erase'] else self.draft['layer']], int(60 + 80 * self.draft['opacity'])))
                pen.setWidthF(max(self.draft['size'] * min(h, w), 1.0))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap); pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.save(); p.setPen(pen)
                if len(pts) == 1:
                    p.drawPoint(pts[0])
                else:
                    path = QPainterPath(pts[0])
                    for q in pts[1:]:
                        path.lineTo(q)
                    p.drawPath(path)
                p.restore()
            if self.tool != 'navigate' and self.cursor_pos is not None:
                rad = max(self.brush['size'] * min(h, w) / 2, 0.5)
                p.save(); p.setRenderHint(QPainter.RenderHint.Antialiasing)
                pen = QPen(QColor('#111111'), 3); pen.setCosmetic(True); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(self.cursor_pos, rad, rad)
                pen = QPen(QColor(*self.colors[self.tool]), 1); pen.setCosmetic(True); p.setPen(pen)
                p.drawEllipse(self.cursor_pos, rad, rad)
                p.restore()

    class PreviewWindow(QWidget):
        """The image pane on the right of the window (shared by all tools): the original before a run, then before / after with the
        same linked auto-stretch; hold Space (or Show original) for the original; zoom / pan like Umbra Guided GHS."""
        def __init__(self, spec=None, embedded=True):
            super().__init__()
            self.spec, self.embedded = spec, True
            self.pix = {}
            self.size_note = ''
            self.titles = {'before': 'Original', 'after': 'Result', 'alt': 'Gradient model'}
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 12, 14, 12)
            lay.setSpacing(6)
            row = QHBoxLayout()
            row.setSpacing(6)
            self.caption = QLabel('', objectName='Badge')
            row.addWidget(self.caption)
            row.addStretch(1)
            self.alt_btn = QPushButton('Show gradient model')
            self.alt_btn.setCheckable(True)
            self.alt_btn.setToolTip('Shows the gradient model: the background that is subtracted. Press again to go back to the result.')
            self.alt_btn.toggled.connect(lambda c: (self.hold(False) if c else None, self.refresh()))
            self.alt_btn.hide()
            row.addWidget(self.alt_btn)
            self.fit_btn = QPushButton('Fit', objectName='ZoomButton')
            self.fit_btn.setToolTip('Fit the whole image in the window (or double-click the image).')
            self.fit_btn.clicked.connect(lambda: self.view.fit())
            row.addWidget(self.fit_btn)
            self.orig_btn = QPushButton('Show original')
            self.show_orig = False
            self.orig_btn.setToolTip('Hold the button (or the Space key) to see the original; release to go back to the result.')
            self.orig_btn.pressed.connect(lambda: self.hold(True))
            self.orig_btn.released.connect(lambda: self.hold(False))
            self.orig_btn.setEnabled(False)
            row.addWidget(self.orig_btn)
            lay.addLayout(row)
            self.view = ZoomView(self)
            lay.addWidget(self.view, 1)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def set_alt(self, label):
            """label: text of the alternative-view button (e.g. 'Show gradient model'), or None for tools without one."""
            self.alt_label = label
            if label:
                self.alt_btn.setText(label)
                self.titles['alt'] = label[5:].capitalize() if label.startswith('Show ') else label
            if not label:
                self.alt_btn.setChecked(False)
                self.alt_btn.hide()

        def hold(self, flag):
            self.show_orig = bool(flag) and self.orig_btn.isEnabled()
            self.refresh()

        def show_state(self, state):
            """state: dict saved by a tool page (see snapshot), or None for an empty pane."""
            if not state:
                self.pix = {}
                self.orig_btn.setEnabled(False)
                self.alt_btn.setChecked(False); self.alt_btn.hide()
                self.caption.setText('')
                self.view.clear('No image loaded in Siril.')
                return
            self.pix = dict(state['pix'])
            self.size_note = state.get('note', '')
            self.orig_btn.setEnabled(state['result'])
            has_alt = 'alt' in self.pix and bool(getattr(self, 'alt_label', None))
            self.alt_btn.setVisible(has_alt)
            if not has_alt:
                self.alt_btn.setChecked(False)
            self.show_orig = False
            self.caption.setText(state['caption'])
            self.refresh(keep_view=state.get('keep_view', True))

        def show_result(self, before, after, alt, caption):
            """before / after / alt: (H,W,3) uint8 (alt may be None)."""
            st = dict(pix={'before': QPixmap.fromImage(qimage(before)), 'after': QPixmap.fromImage(qimage(after))}, result=True, caption=caption,
                      note=self.size_note, keep_view=False)
            if alt is not None:
                st['pix']['alt'] = QPixmap.fromImage(qimage(alt))
            self.show_state(st)
            return st

        def show_original(self, before, caption):
            """Before any run: the original only."""
            p = QPixmap.fromImage(qimage(before))
            st = dict(pix={'before': p, 'after': p}, result=False, caption=caption, note='', keep_view=False)
            self.show_state(st)
            return st

        def current_key(self):
            if self.alt_btn.isVisible() and self.alt_btn.isChecked() and 'alt' in self.pix:
                return 'alt'
            return 'before' if self.show_orig else 'after'

        def refresh(self, keep_view=True):
            if len(self.pix) > 1 and self.orig_btn.isEnabled():
                self.caption.setText(self.titles[self.current_key()] + self.size_note)
            p = self.pix.get(self.current_key())
            if p is None or p.isNull():
                return
            self.view.set_pixmap(p, keep_view=keep_view)

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(True)
            else:
                super().keyPressEvent(e)

        def keyReleaseEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.hold(False)
            else:
                super().keyReleaseEvent(e)

    class Ui:
        """Control builders handed to the tool panels so every tool looks the same."""
        def __init__(self, view):
            self.view = view

        def slider(self, layout, text, lo, hi, default, fmt, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            val = QLabel(fmt(default), objectName='Value')
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val)
            layout.addLayout(row)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi); s.setValue(default)
            s.valueChanged.connect(lambda v, l=val: l.setText(fmt(v)))
            if tip:
                label.setToolTip(tip); s.setToolTip(tip); val.setToolTip(tip)
            layout.addWidget(s)
            s.label_widget, s.value_widget = label, val
            return s

        def check(self, layout, text, tip=None, checked=False):
            c = QCheckBox(text)
            c.setChecked(checked)
            if tip:
                c.setToolTip(tip)
            layout.addWidget(c)
            return c

        def combo(self, layout, text, items, tip=None):
            row = QHBoxLayout()
            label = QLabel(text)
            row.addWidget(label)
            cb = ModernComboBox()
            for t, v in items:
                cb.addItem(t, v)
            if tip:
                label.setToolTip(tip); cb.setToolTip(tip)
            row.addWidget(cb, 1)
            layout.addLayout(row)
            cb.label_widget = label
            return cb

        def note(self, layout, text):
            l = QLabel(text, objectName='Note', wordWrap=True)
            layout.addWidget(l)
            return l

        @staticmethod
        def show(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setVisible(bool(flag))
            QTimer.singleShot(0, lambda: widget.window().adjustSize())

        @staticmethod
        def enable(widget, flag):
            for w in (widget, getattr(widget, 'label_widget', None), getattr(widget, 'value_widget', None)):
                if w is not None:
                    w.setEnabled(bool(flag))

        def model_dir(self):
            return model_dir(self.view.spec)[0]

    class ToolView(QWidget):
        """One tool: title, model card + tool controls, Preview / Apply, progress, status, Close. Used alone or as a page of the combined window."""
        def __init__(self, spec, siril=None, fake_image=None, prefs=None, upd=None, show_title=True, smoke=False, pane=None):
            super().__init__()
            self.spec, self.siril, self.fake, self.prefs, self.smoke = spec, siril, fake_image, prefs, smoke
            self.upd = upd or updspec_for(spec)
            self.worker = self.net = self.preview = None
            self.info = None
            self.job = None
            self.cache = None            # (identity, params key, out, info) of the last whole-image preview (Gradient)
            self.last_preview = None     # (before, after, alt) uint8 of the last preview
            self.result = None
            self.last_error = None
            self.busy = False
            if siril is not None and callable(getattr(spec, 'on_siril', None)):
                try:
                    spec.on_siril(siril)
                except Exception:
                    pass
            saved_model, saved_settings = (prefs.load(spec.key) if prefs else (None, {}))
            self.saved_model = saved_model
            self.left = self                          # the controls column; the image pane on the right belongs to the window
            lay = QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(8)
            self.lay = lay
            if show_title:
                lay.addWidget(QLabel(spec.title, objectName='Title'))
                lay.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
            for n in spec.notes:
                lay.addWidget(QLabel(n, objectName='Note', wordWrap=True))
            card = QWidget(objectName='Card')
            card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 12)
            mrow = QHBoxLayout()
            mrow.addWidget(QLabel('Model'))
            self.model_box = ModernComboBox()
            mrow.addWidget(self.model_box, 1)
            self.dl_btn = QPushButton('Download')        # downloads (or updates) the model selected on the left
            self.dl_btn.setMinimumHeight(28)
            self.dl_btn.setToolTip("Download the selected model from this project's GitHub releases (checked against a SHA-256 before it is installed).")
            self.dl_btn.clicked.connect(self.download_clicked)
            mrow.addWidget(self.dl_btn)
            cl.addLayout(mrow)
            self.upd_note = QLabel('', objectName='Note', wordWrap=True)   # result of the update check made at every start
            self.upd_note.setOpenExternalLinks(True)
            self.upd_note.setTextFormat(Qt.TextFormat.RichText)
            self.upd_note.hide()
            cl.addWidget(self.upd_note)
            self.ui = Ui(self)
            self.panel = spec.build_panel(self.ui, cl, self)
            self.panel.set_settings(saved_settings)
            lay.addWidget(card)
            brow2 = QHBoxLayout()
            self.preview_btn = QPushButton('Preview')
            self.preview_btn.setToolTip('Runs on the selection in Siril (or a centred crop) and shows the result here. The image is not changed.' if not spec.preview_whole
                                        else 'Runs on the whole image and shows the result here, downsampled. The image is not changed.')
            self.preview_btn.clicked.connect(lambda: self.start('preview'))
            self.apply_btn = QPushButton('Apply', objectName='Apply')
            self.apply_btn.clicked.connect(lambda: self.start('apply'))
            if spec.inline_preview:                   # Apply shows the result in the window; Apply to Siril (beside Close) writes it
                self.preview_btn.setText('Apply')
                self.preview_btn.setToolTip('Runs on the whole image and shows the result on the right. The image in Siril is not changed.')
                self.apply_btn.setText('Apply to Siril')
                self.apply_btn.setToolTip('Writes the result to the image in Siril (Undo available). Reuses the result shown on the right when nothing has changed.')
                brow2.addWidget(self.preview_btn)
            else:
                brow2.addWidget(self.preview_btn); brow2.addWidget(self.apply_btn)
            self.edit_btn = None
            if spec.has_editor:                       # Starless: Keep / Remove brushes in their own window
                self.edit_btn = QPushButton('Edit marks')
                self.edit_btn.setToolTip('Opens the editor for correction marks (Remove / Keep brushes) on the whole image.')
                self.edit_btn.clicked.connect(lambda: self.start('edit'))
                brow2.addWidget(self.edit_btn)
            lay.addLayout(brow2)
            prow = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setRange(0, 0); self.progress.setTextVisible(False); self.progress.hide()
            self.cancel_btn = QPushButton('Cancel')
            self.cancel_btn.clicked.connect(self.cancel)
            self.cancel_btn.hide()
            prow.addWidget(self.progress, 1); prow.addWidget(self.cancel_btn)
            lay.addLayout(prow)
            self.idle_msg = IDLE_SIRIL if siril is not None else IDLE_TEST
            self.status = QLabel(self.idle_msg, objectName='Note', wordWrap=True)
            lay.addWidget(self.status)
            close = QPushButton('Close')
            close.clicked.connect(lambda: self.window().close())
            self.undo_btn = QPushButton('Undo all')
            self.undo_btn.setToolTip('Puts back the image as it was when this window opened (Siril Undo still works afterwards).')
            self.undo_btn.clicked.connect(self.undo_all)
            self.undo_btn.setEnabled(False)
            if spec.inline_preview:
                crow = QHBoxLayout()
                crow.addWidget(close); crow.addWidget(self.undo_btn); crow.addWidget(self.apply_btn)
                lay.addLayout(crow)
            else:
                lay.addWidget(close)
            self.idle_msg = (IDLE_INLINE if siril is not None else IDLE_TEST) if spec.inline_preview else self.idle_msg
            self.status.setText(self.idle_msg)
            self.model_box.currentIndexChanged.connect(lambda *_: self.model_changed())
            self.folder_note = None
            lay.addStretch(1)
            self.preview = pane                       # shared image pane (MainWindow)
            self.pane_state = None                    # what this page last showed in the pane: (image identity, state)
            self.refresh_models()

        def activate(self):
            """This page became the current one: show its last result in the pane if it belongs to the current image, else the original."""
            w = self.window()
            if self.preview is None:
                return
            self.preview.set_alt(self.spec.alt[0] if self.spec.alt else None)
            hook = getattr(self.panel, 'pane_activated', None)
            if hook is not None:
                hook(self.preview)
            else:
                self.preview.view.set_brush(None)
            if self.pane_state is not None and self.pane_state[0] == getattr(w, 'orig_identity', None):
                self.preview.show_state(dict(self.pane_state[1], keep_view=True))
            else:
                getattr(w, 'show_original', lambda: None)()

        # ---- status
        def set_status(self, msg, error=False):
            self.status.setText(msg)
            self.status.setStyleSheet('color: #e05050;' if error else '')
            if self.isVisible():                      # a wrapped status line may need more (or less) height
                QTimer.singleShot(0, self.window().adjustSize)

        def model_path(self):
            d = self.model_box.currentData()
            return d if isinstance(d, str) else None      # a '(Not Downloaded)' entry holds its manifest entry instead of a path

        def selected_entry(self):
            """(manifest entry, state) for the selected dropdown item, state 'missing' / 'update' / 'current', or (None, None)."""
            d = self.model_box.currentData()
            if isinstance(d, dict):
                return d, 'missing'
            if isinstance(d, str) and self.info:
                for m, st in self.info['models']:
                    if not m.get('kind') and Path(d).name == m['file'] and Path(d).parent == Path(download_dir(self.spec)):
                        return m, st
            return None, None

        def other_downloads(self):
            """Manifest items other than the selected model that can be installed or updated (e.g. the Auto-keep model)."""
            if not self.info:
                return []
            sel = self.selected_entry()[0]
            offer = getattr(self.spec, 'offer_extras', True)
            return [(m, st) for m, st in self.info['models'] if st != 'current' and m is not sel and ((m.get('kind') and offer) or (st == 'update' and not m.get('kind')))]

        def update_dl_button(self, busy=None):
            busy = self.busy if busy is None else busy
            e, st = self.selected_entry()
            if st == 'missing':
                self.dl_btn.setText('Download'); self.dl_btn.setEnabled(not busy)
                self.dl_btn.setToolTip(f"Download {e['name']} ({e['size_bytes'] / 1e6:.0f} MB). Checked against a SHA-256 before it is installed.")
            elif st == 'update':
                self.dl_btn.setText('Update'); self.dl_btn.setEnabled(not busy)
                self.dl_btn.setToolTip(f"Download the published version of {e['name']} ({e['size_bytes'] / 1e6:.0f} MB).")
            else:
                more = self.other_downloads()
                self.dl_btn.setText('Download'); self.dl_btn.setEnabled(bool(more) and not busy)
                self.dl_btn.setToolTip('Other files to download: ' + ', '.join(m['name'] for m, _ in more) if more
                                       else 'The selected model is installed.')
            hot = st in ('missing', 'update') or not self.model_box.count()
            self.dl_btn.setStyleSheet('background: #1274d5; color: #ffffff; font-weight: bold;' if hot and not busy else '')

        def save_prefs(self):
            if self.prefs:
                self.prefs.save(self.spec.key, self.model_path(), self.panel.settings(), getattr(self.window(), 'current_key', None))

        def set_busy(self, flag, net=False):
            self.busy = flag
            has = bool(self.model_path())
            self.preview_btn.setEnabled(not flag and has); self.apply_btn.setEnabled(not flag and has)
            self.update_dl_button(busy=flag)
            if getattr(self, 'edit_btn', None) is not None:
                self.edit_btn.setEnabled(not flag and has)
            if getattr(self, 'undo_btn', None) is not None:
                self.undo_btn.setEnabled(not flag and bool(getattr(self.window(), 'applied', False)))
            self.model_box.setEnabled(not flag)
            self.progress.setVisible(flag); self.cancel_btn.setVisible(flag and not net)
            if flag:
                self.progress.setRange(0, 0)

        # ---- models
        def refresh_models(self, select=None):
            sp = self.spec
            cur = select or self.model_path() or default_model_path(sp, self.saved_model)
            self.model_box.blockSignals(True)
            self.model_box.clear()
            items = list_models(sp)
            for label, path in items:
                self.model_box.addItem(label, path)
            i = self.model_box.findData(cur) if cur else -1
            if i < 0 and cur and Path(cur).is_file():
                self.model_box.addItem(model_label(sp, Path(cur).stem), cur)
                i = self.model_box.count() - 1
            n_installed = self.model_box.count()
            for m, st in (self.info or {}).get('models', []):    # published models not installed yet (the label loses the tag once installed)
                if st == 'missing' and not m.get('kind'):
                    self.model_box.addItem(f"{model_label(sp, Path(m['file']).stem)} (Not Downloaded)", m)
                    if i < 0 and not n_installed and m.get('default'):
                        i = self.model_box.count() - 1
            if self.model_box.count():
                self.model_box.setCurrentIndex(max(i, 0))
            self.model_box.blockSignals(False)
            has = bool(self.model_path())
            self.preview_btn.setEnabled(has and not self.busy); self.apply_btn.setEnabled(has and not self.busy)
            self.update_dl_button()
            d, fb = model_dir(sp)
            if fb and self.folder_note is None:
                self.folder_note = f"Using models from the old '{sp.fallback_folder}' folder."
            self.model_changed(remember=bool(select))

        def model_changed(self, *_, remember=False):
            path = self.model_path()
            self.update_dl_button()
            has = bool(path)
            self.preview_btn.setEnabled(has and not self.busy); self.apply_btn.setEnabled(has and not self.busy)
            if not path:
                self.panel.model_changed(None)
                e, st = self.selected_entry()
                if st == 'missing':
                    self.set_status(f"{e['name']} is not downloaded yet. Click Download to install it ({e['size_bytes'] / 1e6:.0f} MB).")
                else:
                    self.set_status(missing_model_message(self.spec, self.info))
                return
            self.panel.model_changed(path)
            if remember:
                self.save_prefs()
            idle = self.idle_msg
            if self.folder_note:
                idle = self.folder_note + ' ' + idle
            self.set_status(idle)

        # ---- updates / downloads (a button press starts every network action, except the manifest lookup at launch)
        def run_net(self, fn, on_done, busy_text):
            if self.net is not None and self.net.isRunning():
                return
            self.net = Task(fn)
            self.net.prog.connect(self.net_progress)
            self.net.done.connect(lambda r: (self.set_busy(False), on_done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), self.net_failed(m)))
            self.set_busy(True, net=True)
            self.set_status(busy_text)
            self.net.start()

        def net_progress(self, got, total):
            if total > 0:
                self.progress.setRange(0, 1000); self.progress.setValue(int(1000 * got / total))
                self.set_status(f'Downloading… {got / 1e6:.0f} / {total / 1e6:.0f} MB')

        def net_failed(self, msg):
            self.set_status(msg + ('' if self.model_path() else ' No model is installed yet.'), True)

        def fetch_info(self, manual=False, then=None):
            sp = self.spec
            if offline_mode(sp):
                self.upd_note.setText(f'Offline mode ({sp.prefix}OFFLINE is set); using the installed models.'); self.upd_note.show()
                return
            def work(prog):
                return check_updates(fetch_manifest(self.upd, use_cache=not manual), sp)
            def done(res):
                self.info = res
                self.show_update_note(res)
                keep = self.model_box.currentData()
                self.refresh_models(select=keep if isinstance(keep, str) else None)
                if then:
                    then()
            def failed(msg):
                self.info = None
                self.upd_note.setStyleSheet('')
                self.upd_note.setText(msg); self.upd_note.show()
                self.set_status(msg if not self.model_path() else self.idle_msg, not self.model_path())
            self.net = Task(work)
            self.net.done.connect(lambda r: (self.set_busy(False), done(r)))
            self.net.failed.connect(lambda m: (self.set_busy(False), failed(m)))
            self.set_busy(True, net=True)
            self.set_status('Checking the model list…')
            self.net.start()

        def show_update_note(self, res):
            """Green when the script and this tool's models are up to date, yellow when something new is available."""
            from html import escape
            lines = []
            if res['script_newer']:
                url = res['script_url']
                lines.append(f'There is an update to this script available ({escape(res["script_version"])}, you have {VERSION}).'
                             + (f' <a href="{escape(url)}" style="color: #e8c547;">Release page</a>' if url else ''))
            if any(st == 'missing' and not m.get('kind') for m, st in res['models']):
                lines.append('There is a new model available for this tool, select the model in the dropdown and click Download to install.')
            if any(st == 'update' and not m.get('kind') for m, st in res['models']):
                lines.append('An update is available for a model you have: select it in the dropdown and click Update.')
            if lines:
                self.upd_note.setStyleSheet('color: #e8c547;')
                self.upd_note.setText('<br>'.join(lines))
            else:
                self.upd_note.setStyleSheet('color: #5cc96b;')
                self.upd_note.setText('The models for this tool are up to date.')
            self.upd_note.show()

        def download_clicked(self):
            e, st = self.selected_entry()
            if st in ('missing', 'update'):
                self.start_download(e)
                return
            more = self.other_downloads()
            if not more:
                self.update_dl_button()
                return
            if len(more) == 1:
                self.start_download(more[0][0])
                return
            menu = QMenu(self)
            for m, st2 in more:
                tag = 'Update ' if st2 == 'update' else 'Download '
                menu.addAction(f"{tag}{m['name']} ({m['size_bytes'] / 1e6:.0f} MB)").triggered.connect(lambda _=False, e=m: self.start_download(e))
            menu.exec(self.dl_btn.mapToGlobal(self.dl_btn.rect().bottomLeft()))

        def show_dl_menu(self):
            self.download_clicked()

        def start_download(self, entry):
            def work(prog):
                return download_model(entry, self.upd, download_dir(self.spec), progress=prog)
            def done(path):
                if self.info:                       # the downloaded file now matches the published one
                    self.info = dict(self.info, models=[(m, 'current' if m is entry else st) for m, st in self.info['models']])
                    self.show_update_note(self.info)
                if entry.get('kind'):               # an extra file (e.g. the Auto-keep model): not a model for the picker
                    getattr(self.panel, 'extras_changed', lambda: None)()
                    self.update_dl_button()
                    self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
                    return
                self.refresh_models(select=str(path))
                self.set_status(f"Installed {entry['name']}. " + self.idle_msg)
            self.run_net(work, done, f"Downloading {entry['name']}…")

        # ---- run
        def cancel(self):
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel_flag = True
                self.cancel_btn.setEnabled(False)
            hook = getattr(self.panel, 'cancel', None)
            if hook is not None:
                hook()

        def undo_all(self):
            """Write back the image this window started with (keeps a Siril Undo state, so this can be undone too)."""
            w = self.window()
            start = getattr(w, 'start_image', None)
            if start is None or self.busy:
                return
            data, meta = start
            try:
                if self.siril is not None:
                    ident = write_result(self.siril, data, meta, w.orig_identity, f'{self.spec.name} {VERSION} undo all')
                    self.siril.log(f'{self.spec.name}: Undo all - the image is back to how it was when the window opened.')
                else:
                    ident = fingerprint(data)
            except Exception as e:
                self.set_status(f'Undo all failed: {e}', True)
                return
            for v in w.views.values():
                v.cache = None; v.pane_state = None; v.result = None
            w.note_image(ident, data, show=True)
            w.applied = False
            for v in w.views.values():
                v.undo_btn.setEnabled(False)
            self.set_status('Undo all: the image is back to how it was when the window opened.')

        def read_image(self, kind):
            """-> (data, meta, identity, header_text, region note, rect)"""
            sp = self.spec
            if self.siril is not None:
                with self.siril.image_lock():
                    raw = read_pixels(self.siril)
                    data, meta = decode_pixels(raw)
                    identity = fingerprint(raw)
                    htxt = siril_header_text(self.siril)
            else:
                data, meta, identity, htxt = self.fake, dict(dtype=np.dtype('float32'), shape=self.fake.shape), fingerprint(self.fake), ''
            return data, meta, identity, htxt

        def start(self, kind):
            if self.busy:
                return
            if getattr(self.panel, 'editor_busy', lambda: False)():
                self.set_status('The preview window is still working. Wait until it has finished.')
                return
            sp = self.spec
            path = self.model_path()
            if not path:
                self.set_status(missing_model_message(sp, self.info), True)
                return
            try:
                data, meta, identity, htxt = self.read_image(kind)
            except Exception as e:
                self.set_status(f'Could not read the image: {e}', True)
                return
            params = dict(self.panel.params())
            params.update(sp.image_params(htxt))
            if kind == 'edit' and sp.has_editor:
                self.save_prefs()
                self.panel.open_editor(self, data, meta, identity, params, path)
                return
            rect, region, crop = None, 'whole image', None
            full = data
            if kind == 'preview':
                params = sp.preview_params(params)
                if not sp.preview_whole:
                    sel = region_from_selection(self.siril) if self.siril is not None else None
                    H, W = full.shape[-2:]
                    # Siril selections are in display coordinates; pixel rows are bottom-up (rows H-y-h .. H-y)
                    _, arect = crop_region(full, None if sel is None else (sel[0], H - sel[1] - sel[3], sel[2], sel[3]))
                    rect = sel if sel is not None else arect
                    x, y, w, h = arect
                    data = np.ascontiguousarray(full[..., y:y + h, x:x + w])
                    if w < 512 or h < 512:                   # fixed 512 px models: give small regions context, crop back afterwards
                        tw, th = min(max(w, 640), W), min(max(h, 640), H)
                        x0 = min(max(x - (tw - w) // 2, 0), W - tw); y0 = min(max(y - (th - h) // 2, 0), H - th)
                        if (tw, th) != (w, h):
                            data = np.ascontiguousarray(full[..., y0:y0 + th, x0:x0 + tw])
                            crop = (x - x0, y - y0, w, h)
                    region = (f'selection {rect[2]}×{rect[3]} px at {rect[0]},{rect[1]}' if sel is not None
                              else (f'centre crop {rect[2]}×{rect[3]} px' if (rect[2], rect[3]) != (W, H) else f'whole image {W}×{H} px'))
                else:
                    region = f'whole image {data.shape[-1]}×{data.shape[-2]} px'
            key = json.dumps(params, sort_keys=True, default=str) + str(path)
            if kind == 'apply' and sp.reuse_preview and self.cache and identity is not None and self.cache[0] == identity and self.cache[1] == key:
                self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt)
                self.finish(self.cache[2], self.cache[3], 0.0, self.cache[4], self.cache[5], reused=True)
                return
            if sp.linear_only:
                w = stretched_warning(data)
                if w:
                    msg = f'{sp.name}: Warning: {w}'
                    if self.siril is not None:
                        try:
                            self.siril.log(msg)
                        except Exception:
                            pass
                    else:
                        print(msg, flush=True)
            self.job = dict(kind=kind, meta=meta, identity=identity, params=params, region=region, data=data, key=key, htxt=htxt, rect=rect, crop=crop, full=full)
            try:
                extra = sp.prepare_job(self, self.job)
            except Exception as e:
                self.set_status(str(e), True)
                return
            self.result = None; self.last_error = None
            self.save_prefs()
            self.set_busy(True)
            self.cancel_btn.setEnabled(True)
            self.set_status('Loading model…' if str(path) not in [c[0] for c in _ENGINE_CACHE.values()] else 'Running…')
            self.worker = Worker(sp, path, data, params, extra)
            self.worker.stage.connect(self.set_status)
            self.worker.prog.connect(self.on_progress)
            self.worker.done.connect(self.finish)
            self.worker.failed.connect(self.fail)
            self.worker.cancelled.connect(self.was_cancelled)
            self.worker.start()

        def on_progress(self, n, total):
            if total > 0:
                self.progress.setRange(0, total); self.progress.setValue(n)
                self.set_status(f'Running… {self.spec.progress_unit} {n} / {total}')

        def was_cancelled(self):
            self.set_busy(False)
            self.set_status('Cancelled. The image was not changed.')

        def fail(self, msg, loading=False):
            self.set_busy(False)
            self.last_error = msg
            self.set_status((f'Could not load the model: {msg}' if loading else msg), True)

        def finish(self, out, info, secs, provider, note, reused=False):
            sp, job = self.spec, self.job
            self.set_busy(False)
            extra = ' '.join(dict.fromkeys(info.get('notes') or []))
            if note and note not in extra:
                extra = (extra + ' ' + note).strip()
            try:
                if job['kind'] == 'preview':
                    if job.get('crop'):                       # run on a larger context region; show only the selected part
                        cx, cy, cw, ch = job['crop']
                        sl = (Ellipsis, slice(cy, cy + ch), slice(cx, cx + cw))
                        out = out[sl]
                        job = dict(job, data=job['data'][sl])
                        info = dict(info, **{k: np.asarray(v)[sl] for k, v in info.items() if isinstance(v, np.ndarray) and v.ndim >= 2})
                    self.show_preview(out, info, job)
                    QTimer.singleShot(0, getattr(self.panel, 'after_run', lambda: None))
                    if sp.reuse_preview and job['identity'] is not None:
                        self.cache = (job['identity'], job['key'], out, info, provider, note)
                    self.result = out
                    tail = (' Press Apply to Siril to write it.' if sp.inline_preview else ' The image was not changed.')
                    self.set_status(f'{"Done" if sp.inline_preview else "Preview done"} in {secs:.1f} s. Ran on {provider}.' + tail + (' ' + extra if extra else ''))
                    return
                label = f"{sp.name} {VERSION} {sp.label(job['params'])}"
                write_out = sp.output(job['params'], out, info)
                msg = ''
                ctx = types.SimpleNamespace(siril=self.siril, header_text=job['htxt'], history=label, view=self)
                if self.siril is not None:
                    self.cache = None
                    self.job['identity'] = write_result(self.siril, write_out, job['meta'], job['identity'], label)
                    getattr(self.window(), 'note_image', lambda *a, **k: None)(self.job['identity'], write_out, show=True)   # the pane shows the new image
                    getattr(self.window(), 'mark_applied', lambda: None)()
                    msg = sp.post_apply(ctx, job['params'], info) or ''
                    self.siril.log(f'{label} done in {secs:.1f} s (Undo available).' + (' ' + msg if msg else ''))
                H0, W0 = job['data'].shape[-2:]
                if not sp.preview_whole and min(H0, W0) < 512:
                    extra = (extra + ' Images smaller than 512 px give less accurate results near the edges.').strip()
                self.result = write_out
                self.set_status(f'Done in {secs:.1f} s. Ran on {provider}. ' + ('Applied to Siril; use Siril Undo to revert.' if self.siril is not None else 'Test mode: nothing was written.')
                                + (' ' + msg if msg else '') + (' ' + extra if extra else ''))
            except Exception as e:
                self.fail(str(e))

        def show_preview(self, out, info, job):
            sp = self.spec
            before = pool_for_display(job['data'], PANE_SIDE)
            after = pool_for_display(out, PANE_SIDE)
            flip = self.siril is not None
            sp_before = stretch_params(before)
            b8, a8 = to_rgb8(before, sp_before, flip), to_rgb8(after, sp_before, flip)
            alt8 = None
            if sp.alt and info.get(sp.alt[1]) is not None:
                alt = pool_for_display(np.asarray(info[sp.alt[1]], np.float32), PANE_SIDE)
                alt8 = to_rgb8(alt, sp_before if getattr(sp, 'alt_same_stretch', False) else stretch_params(alt), flip)
            self.last_preview = (b8, a8, alt8)
            w = self.window()
            getattr(w, 'note_image', lambda *a: None)(job.get('identity'), job.get('full'))
            self.preview.set_alt(sp.alt[0] if sp.alt else None)
            self.preview.size_note = f' · {job["data"].shape[-1]}×{job["data"].shape[-2]} px'
            st = self.preview.show_result(b8, a8, alt8, f'Preview of the {job["region"]}. Same auto-stretch for before and after.')
            self.pane_state = (getattr(w, 'orig_identity', job.get('identity')), st)

        def shutdown(self):
            for t in (self.worker, self.net):
                if t is not None and t.isRunning():
                    if t is self.worker:
                        t.cancel_flag = True
                    t.wait(8000)
            getattr(self.panel, 'shutdown', lambda: None)()

    class MainWindow(QWidget):
        """Like Umbra Guided GHS: a fixed-width controls column on the left (scrolls if a tool needs more room) and one zoomable image
        pane on the right that shows the original until a tool has run.  The window keeps its size when you switch tools or models."""
        LEFT_W = 400

        def __init__(self, specs, combined, siril=None, fake_image=None, smoke=False):
            super().__init__()
            self.specs = {s.key: s for s in specs}
            self.combined = combined
            self.siril, self.fake = siril, fake_image
            self.setStyleSheet(STYLE)
            self.views = {}
            self.orig_identity, self.orig_state = None, None
            self.start_image, self.applied = None, False
            outer = QHBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            left = QWidget()
            left.setFixedWidth(self.LEFT_W)
            ll = QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            ll.setSpacing(0)
            content = QWidget()
            lay = QVBoxLayout(content)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            scroll = QScrollArea()
            scroll.setWidget(content)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            ll.addWidget(scroll)
            outer.addWidget(left)
            self.pane = PreviewWindow()
            outer.addWidget(self.pane, 1)
            if combined:
                self.setWindowTitle('Umbra Astro AI (BETA)')
                prefs = Prefs(SCRIPT_DIR / 'Umbra_Astro_AI_prefs.json', combined=True, enabled=not smoke)
                top = QWidget()
                tl = QVBoxLayout(top)
                tl.setContentsMargins(14, 12, 14, 0)
                tl.setSpacing(4)
                tl.addWidget(QLabel('Umbra Astro AI (BETA)', objectName='Title'))
                tl.addWidget(QLabel(f'v{VERSION}', objectName='Badge'))
                row = QHBoxLayout()
                row.addWidget(QLabel('Tool'))
                self.tool_box = ModernComboBox()
                for s in specs:
                    self.tool_box.addItem(s.name, s.key)
                row.addWidget(self.tool_box, 1)
                tl.addLayout(row)
                lay.addWidget(top)
                self.top = top
                self.stack = None
                for s in specs:
                    v = ToolView(s, siril, fake_image, prefs, combined_upd(specs), show_title=False, smoke=smoke, pane=self.pane)
                    self.views[s.key] = v
                    lay.addWidget(v)
                    v.hide()
                self.prefs = prefs
                first = prefs.load_tool()
                self.current_key = first if first in self.views else specs[0].key
                self.tool_box.setCurrentIndex(self.tool_box.findData(self.current_key))
                self.tool_box.currentIndexChanged.connect(lambda *_: self.switch(self.tool_box.currentData()))
            else:
                s = specs[0]
                self.setWindowTitle(s.title)
                self.current_key = s.key
                v = ToolView(s, siril, fake_image, Prefs(SCRIPT_DIR / f'{s.script}_prefs.json', enabled=not smoke), show_title=True, smoke=smoke, pane=self.pane)
                self.views[s.key] = v
                lay.addWidget(v)
            # one width for every tool page (the widest one), so switching tools never moves the image pane
            need = max([v.minimumSizeHint().width() for v in self.views.values()] + [self.LEFT_W]) + 10
            left.setFixedWidth(need)
            self.left_w = need
            avail = QApplication.primaryScreen().availableGeometry()   # fit the screen, like Umbra Guided GHS
            self.resize(min(1400, avail.width() - 40), min(900, avail.height() - 40))
            self.setMinimumSize(self.left_w + 360, 520)
            self.switch(self.current_key, initial=True)
            QTimer.singleShot(0, self.load_original)

        def view(self):
            return self.views[self.current_key]

        def adjustSize(self):
            """Tool pages ask for a refit when rows appear / disappear or a status wraps: the window keeps its size (the left column scrolls)."""
            if self.layout() is not None:
                self.layout().activate()

        def refit(self):
            self.adjustSize()

        # ---- the original image in the pane
        def load_original(self):
            """Read the image (from Siril, or the test image) and show it in the pane, auto-stretched."""
            try:
                data, meta, identity, htxt = self.view().read_image('preview')
            except Exception:
                self.orig_identity, self.orig_state = None, None
                self.pane.show_state(None)
                return
            if getattr(self, 'start_image', None) is None:          # for Undo all
                self.start_image = (np.array(data, copy=True), meta)
            self.note_image(identity, data, show=True)
            self.view().activate()                                  # brushes etc. for the first page

        def mark_applied(self):
            self.applied = True
            for v in self.views.values():
                v.undo_btn.setEnabled(not v.busy)

        def note_image(self, identity, data, show=False):
            """Remember the current image (identity + display copy). show: put it in the pane now."""
            if identity is None or data is None:
                return
            if identity != self.orig_identity or self.orig_state is None:
                full = np.asarray(data, np.float32)
                b = pool_for_display(full, PANE_SIDE)
                b8 = to_rgb8(b, stretch_params(b), self.siril is not None)
                p = QPixmap.fromImage(qimage(b8))
                self.orig_identity = identity
                self.orig_state = dict(pix={'before': p, 'after': p}, result=False, caption=f'Original · {full.shape[-1]}×{full.shape[-2]} px', note='')
                self.check_image(full)
            if show:
                self.show_original()

        def check_image(self, full):
            """Tool pages that adapt to the image (Deblur: no stars -> Automatic PSF greyed out) check it in the background."""
            self._checks = [t for t in getattr(self, '_checks', []) if t.isRunning()]
            for v in self.views.values():
                fn, cb = getattr(v.panel, 'image_check', None), getattr(v.panel, 'image_checked', None)
                if fn is None or cb is None:
                    continue
                t = Task(lambda prog, fn=fn: fn(full))
                t.done.connect(lambda r, cb=cb: cb(r))
                t.failed.connect(lambda m: None)
                self._checks.append(t)
                t.start()

        def show_original(self):
            if self.orig_state is None:
                self.pane.show_state(None)
                return
            self.pane.set_alt(None)
            self.pane.show_state(dict(self.orig_state, keep_view=True))

        def switch(self, key, initial=False):
            if not initial:
                self.view().save_prefs()
                unload_engines()
            self.current_key = key
            for k, v in self.views.items():
                v.setVisible(k == key)
            self.adjustSize()
            if not initial:
                self.views[key].activate()
                self.view().save_prefs()

        def closeEvent(self, e):
            for t in getattr(self, '_checks', []):
                t.wait(10000)
            for v in self.views.values():
                v.shutdown()
            try:
                self.view().save_prefs()
            except Exception:
                pass
            super().closeEvent(e)

    _QT['ns'] = types.SimpleNamespace(**{k: v for k, v in locals().items() if k in (
        'Worker', 'Task', 'PreviewWindow', 'ZoomView', 'Ui', 'ToolView', 'MainWindow', 'QApplication', 'QTimer', 'ModernComboBox', 'qimage')})
    return _QT['ns']


def fake_image(seed=0, size=192):
    """Synthetic linear RGB frame (noise + a few stars + a gentle gradient) for --ui-test."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    img = (0.02 + 0.003 * rng.standard_normal((3, size, size)) + (0.01 * xx / size)[None]).astype(np.float32)
    for cy, cx in ((40, 50), (110, 120), (150, 40), (70, 150)):
        img += (0.3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2)))[None].astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def run_app(specs, combined, siril=None, fake=None, smoke=False, shots=None):
    """Open the window. smoke: run Preview once for every tool on the fake image, report, exit (return code 0 = all fine)."""
    q = _qt()
    app = q.QApplication.instance() or q.QApplication(sys.argv)
    win = q.MainWindow(specs, combined, siril, fake, smoke)
    ts = os.environ.get('UMBRA_TEST_WINDOW', '')          # tests only: window size 'WxH'
    if smoke and re.fullmatch(r'\d+x\d+', ts):
        win.resize(*map(int, ts.split('x')))
    win.show()
    if not smoke:
        for v in win.views.values():
            if v is win.view():
                q.QTimer.singleShot(200, lambda v=v: (v.fetch_info(manual=False) if not offline_mode(v.spec) else None))
        if combined:
            win.tool_box.currentIndexChanged.connect(lambda *_: (win.view().info is None and not offline_mode(win.view().spec)) and win.view().fetch_info(manual=False))
        return app.exec()
    state = dict(i=0, ok=True, t0=time.time())
    keys = [s.key for s in specs]
    outdir = Path(shots) if shots else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    def snap(v, key):
        if not outdir:
            return
        pre = f'Umbra_Astro_AI_{key}' if combined else win.view().spec.script
        win.grab().save(str(outdir / f'{pre}_panel.png'))
        ed = getattr(v.panel, 'ed', None)
        if ed is not None and ed.isVisible():
            ed.grab().save(str(outdir / f'{pre}_editor.png'))
        if v.preview is not None and v.preview.isVisible():
            v.preview.grab().save(str(outdir / f'{pre}_preview.png'))
            if v.preview.alt_btn is not None and v.preview.alt_btn.isVisible():
                v.preview.alt_btn.setChecked(True)
                v.preview.grab().save(str(outdir / f'{pre}_preview_{v.spec.alt[1]}.png'))
                v.preview.alt_btn.setChecked(False)

    def next_tool():
        if state['i'] >= len(keys):
            print('UI TEST', 'OK' if state['ok'] else 'FAILED', flush=True)
            win.close(); app.quit()
            return
        key = keys[state['i']]
        if combined:
            win.tool_box.setCurrentIndex(win.tool_box.findData(key))
        v = win.view()
        if not v.model_path():
            # no model: Preview/Apply must be disabled and the status must say so
            good = (not v.preview_btn.isEnabled() and not v.apply_btn.isEnabled() and 'No model installed' in v.status.text())
            print(f'UI TEST {key}: no model installed; controls {"OK" if good else "BAD"}: {v.status.text()}', flush=True)
            state['ok'] &= good
            app.processEvents(); snap(v, key)
            state['i'] += 1
            q.QTimer.singleShot(50, next_tool)
            return
        if outdir:                                   # the pane before any run: the original
            app.processEvents()
            win.grab().save(str(outdir / f"{f'Umbra_Astro_AI_{key}' if combined else v.spec.script}_before.png"))
        v.start('preview')
        state['t0'] = time.time()
        q.QTimer.singleShot(250, lambda: poll(v, key))

    def poll(v, key):
        if v.busy and time.time() - state['t0'] < 900:
            q.QTimer.singleShot(250, lambda: poll(v, key))
            return
        ed = getattr(v.panel, 'ed', None)
        ok = v.last_error is None and v.result is not None and bool(np.isfinite(v.result).all()) and v.last_preview is not None
        if ok:
            v.preview.orig_btn.pressed.emit(); ok &= v.preview.current_key() == 'before'; v.preview.orig_btn.released.emit(); ok &= v.preview.current_key() == 'after'
        print(f'UI TEST {key}: {"OK" if ok else "FAILED"} {v.status.text()}', flush=True)
        state['ok'] &= ok
        app.processEvents()
        snap(v, key)
        if ed is not None:
            ed.shutdown()
        state['i'] += 1
        q.QTimer.singleShot(50, next_tool)
    q.QTimer.singleShot(100, next_tool)
    app.exec()
    return 0 if state['ok'] else 1


# ----------------------------------------------------------------------------------------------- CLI
def read_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        hdu = next(x for x in h if x.data is not None)
        data = np.asarray(hdu.data, np.float32)
        header = hdu.header.copy()
        text = ' '.join(repr(x.header) for x in h) + ' ' + ' '.join(str(x.name) for x in h) + ' ' + Path(path).name
    return data, header, text


def run_file(spec, a):
    src, dst = Path(a.image).resolve(), Path(a.out).resolve()
    if dst == src:
        raise SystemExit('Refusing to overwrite the input file (--out equals --image).')
    if dst.exists():
        raise SystemExit(f'Output exists, refusing to overwrite: {dst}')
    data, header, text = read_fits(src)
    params = spec.params_from_args(a)
    params.update(spec.image_params(text))
    if spec.linear_only:
        w = stretched_warning(data)
        if w:
            print(f'{spec.name}: Warning: {w}', flush=True)
    path = a.model or default_model_path(spec)
    if not path:
        raise SystemExit(f'No model installed. Run with --download, or put a .onnx file in {model_dir(spec)[0]}.')
    t0 = time.time()
    eng = get_engine(spec, path)
    out, info = spec.engine_run(eng, data, params, None, None, None)
    label = f"{spec.name} {VERSION} {spec.label(params)}"
    write_out = spec.output(params, out, info)
    for k in ('BSCALE', 'BZERO', 'BLANK'):
        header.pop(k, None)
    header['HISTORY'] = label
    from astropy.io import fits
    fits.PrimaryHDU(np.asarray(write_out, np.float32), header=header).writeto(dst)
    msg = spec.post_cli(a, dst, info, header, label)
    print(f'wrote {dst} shape={write_out.shape} finite={bool(np.isfinite(write_out).all())} device={eng.provider} model={Path(path).name} '
          f'{time.time() - t0:.1f} s' + (f' {msg}' if msg else '') + (f' NOTE: {" ".join(info.get("notes") or [])}' if info.get('notes') else '')
          + (f' NOTE: {eng.note}' if eng.note else ''))


def cli_updates(specs, a, upd_for=None):
    """--check-updates / --download for one or several tools. Returns an exit code."""
    try:
        for spec in specs:
            upd = upd_for(spec) if upd_for else updspec_for(spec)
            if offline_mode(spec):
                print(f'Offline mode ({spec.prefix}OFFLINE is set); nothing to check.')
                continue
            res = check_updates(fetch_manifest(upd), spec)
            if len(specs) > 1:
                print(f'== {spec.name}')
            if a.check_updates:
                print('\n'.join(describe_updates(res, spec)))
            if a.download:
                want = [e for e, st in res['models'] if st != 'current' and (a.download == 'all' or (a.download == 'default' and e.get('default')) or a.download == e['id'])]
                if not want and len(specs) == 1:
                    print('Nothing to download (installed and up to date, or unknown model id).')
                for e in want:
                    last = [0]
                    def pr(g, t, e=e, last=last):
                        if g - last[0] >= 20_000_000 or g == t:
                            last[0] = g; print(f"  {e['name']}: {g / 1e6:.0f} / {t / 1e6:.0f} MB", flush=True)
                    print('installed', download_model(e, upd, download_dir(spec), pr))
        return 0
    except DownloadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


def main(specs, combined=False, doc=''):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', help='standalone: FITS file to process (never modified)')
    ap.add_argument('--out', help='standalone: output FITS (must not exist)')
    ap.add_argument('--model', help='model file (*.onnx); default: the remembered / first model')
    if combined:
        ap.add_argument('--tool', choices=[s.key for s in specs], help='which tool to run / check (default for --check-updates and --download: all)')
    for s in specs:
        s.add_cli(ap)
    ap.add_argument('--check-updates', action='store_true', help='compare installed models and this script with the published manifest, then exit')
    ap.add_argument('--download', nargs='?', const='default', metavar='MODEL_ID|all', help='download a published model (default: the default model) into the models folder, then exit')
    ap.add_argument('--ui-test', action='store_true', help='open the window on a fake image, run Preview once, exit')
    ap.add_argument('--screenshots', help=argparse.SUPPRESS)
    a = ap.parse_args()
    chosen = [s for s in specs if not combined or not a.tool or s.key == a.tool]
    if a.check_updates or a.download:
        if combined:
            return cli_updates(chosen, a, lambda s: combined_upd(specs))
        return cli_updates(chosen, a)
    if a.ui_test:
        return run_app(specs, combined, None, fake_image(), smoke=True, shots=a.screenshots)
    if a.image:
        if not a.out:
            ap.error('--out is required with --image')
        spec = chosen[0] if not combined else (TOOLS.get(a.tool) if a.tool else None)
        if spec is None:
            ap.error('--tool is required with --image in the combined script')
        spec.check_cli(a, ap)
        run_file(spec, a)
        return 0
    try:
        import_engine(specs[0])   # import onnxruntime before Qt starts; any failure is kept and shown in the window
    except Exception:
        pass
    import sirilpy as s
    siril = s.SirilInterface()
    try:
        siril.connect()
        siril.cmd('requires', '1.4.0')
        return run_app(specs, combined, siril)
    finally:
        siril.disconnect()


# ============================================================================================ engines
def _engine_deblur():
    # SPDX-License-Identifier: GPL-3.0-only
    # Umbra Deblur engine, Copyright (c) 2026 Kishan Patel. Own code; no code taken from other scripts.
    """Umbra Deblur engine: pure numpy + scipy + onnxruntime (via a session factory). No Qt, no Siril, no torch.

    Processing code is the Umbra Deconvolve 0.0.14 engine, moved unchanged (the maths is identical).

    Engine(model_path, session_factory).run(data, params, progress=None, cancel=None) -> (out, info)
    params (all optional):
      sharpening      0-8, default 3.0 (models with the Sharpening control: v1.0, v1.1 experimental, legacy fb / fs / umbra_deconvolution_*)
      strength        0-1, default 0.5 (older single-strength models only)
      correct_only    bool, default False: Aberration Removal Only (strength, star and halo forced to 0, sharpening 0)
      fwhm            None = Automatic PSF, else the PSF diameter in pixels
      star, halo      0-1, default 0 (Sharpen Stars / Halo Reduction)
      drizzled        None / bool: True raises the noise estimate for correlated (drizzled) noise
      protect_stars   bool: for fine-detail models, stars come from the v1.0 model (needs star_model_path)
      star_model_path path of the v1.0 model (used when protect_stars and the model is fine-detail)
    info: 'notes' (list of str), 'fwhm' (measured global FWHM, px) and the processing details of the 0.0.14 engine.
    """
    import math
    from pathlib import Path

    import numpy as np
    from scipy.ndimage import (gaussian_filter, maximum_filter, uniform_filter, correlate1d, label, find_objects, binary_dilation)

    TOOL = 'deblur'

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


    def psf_check(y, max_side=3000):
        """Can Automatic PSF measure stars in this image?  y (C,H,W) linear.  Measured on the central max_side x max_side crop (density is per
        megapixel, so a crop gives the same answer, much faster).  -> dict(ok, density (isolated stars per MP), fwhm)."""
        Y = np.nan_to_num(np.asarray(y, np.float32))
        if Y.ndim == 2:
            Y = Y[None]
        if Y.shape[0] == 1:
            Y = np.repeat(Y, 3, 0)
        H, W = Y.shape[-2:]
        y0, x0 = max(0, (H - max_side) // 2), max(0, (W - max_side) // 2)
        Y = np.ascontiguousarray(Y[:, y0:y0 + max_side, x0:x0 + max_side])
        p = measure_psf(Y)
        dens = p['n'] / max(Y.shape[-1] * Y.shape[-2] / 1e6, 1e-6)
        return dict(ok=not (p['gl'] > AUTO_PSF_MAX or dens < STARLESS_DENSITY), density=float(dens), fwhm=float(p['gl']))


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
        """Aberration Removal Only first, then the requested sharpening (single-strength models / Sharpening 0)."""
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



    # ------------------------------------------------------------------------------------------------ ONNX session adapter
    def probe_feed():
        """Random inputs matching the ONNX inputs (x (1,3,512,512), cond (1,6,512,512)), for the GPU-vs-CPU probe."""
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((1, 3, TILE, TILE)) * 2).astype(np.float32)
        c = rng.random((1, 6, TILE, TILE)).astype(np.float32)
        return {'x': x, 'cond': c}


    class OnnxNet:
        """Thin adapter: net(x, cond) -> r (1,3,T,T) through the run function of a session_factory."""

        def __init__(self, path, session_factory):
            self.path = str(path)
            self._run, self.provider, self.note = session_factory(self.path, probe_feed())

        def __call__(self, x, cond):
            return self._run({'x': x, 'cond': cond})[0]


    # ----------------------------------------------------------------------------------------------- glue
    SHARP_DEFAULT = 3.0


    def _stem(model_path):
        return Path(str(model_path or '')).stem.lower()


    def is_fb(model_path):
        """fb models are trained as a small repeatable step (0.65x blur per pass) and use the Sharpening control (0-8 passes)."""
        st = _stem(model_path)
        return (st.startswith('umbra_deblur_v') or st.startswith('umbra_deconv_fb') or st.startswith('umbra_deconv_fs')
                or st.startswith('umbra_deconvolution_'))


    def is_finedetail(model_path):
        st = _stem(model_path)
        return ('finedetail' in st or st.startswith('umbra_deconvolution_1.1') or st.startswith('umbra_deblur_v1.1'))


    def pass_plan(v):
        """Sharpening v (0..8, steps of 0.25) -> list of per-pass strengths after Aberration Removal Only: floor(v) x 1.0, then frac(v) if > 0."""
        v = max(0.0, min(8.0, round(float(v) * 4) / 4))
        n, f = int(v), v - int(v)
        return [1.0] * n + ([round(f, 4)] if f > 1e-6 else [])


    def deconvolve_array(model, data, strength, fwhm, progress=None, star=0.0, halo=0.0, sharpening=None, drizzled=None, finedetail=False, star_model=None):
        """data HxW, CxHxW (C = 1 or 3) or HxWxC (C = 1 or 3) float32 -> (same shape, info). Mono is replicated to 3 channels and averaged back.
        sharpening None (older models): Aberration Removal Only + one sharpen pass at `strength` (0.0.6 behaviour).
        sharpening v (fb models; `strength` ignored): Aberration Removal Only once, then the passes of pass_plan(v) (Sharpen Stars / Halo Reduction on the
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
                                       star=float(star), halo=float(halo))           # older models, or Sharpening 0 (= Aberration Removal Only)
        else:
            P = len(plan)
            def prog(k, n, t):                                                 # progress per pass: (pass, tile) over Aberration Removal Only + P passes
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
            info['notes'] = notes
        out = np.asarray(out, np.float32)
        if mono:
            out = out.mean(0, keepdims=True)
        if layout == 'hw':
            out = out[0]
        elif layout == 'hwc':
            out = np.ascontiguousarray(out.transpose(1, 2, 0))
        return out, info


    class Engine:
        """Loads the model once. provider / note describe the session (GPU or CPU)."""

        def __init__(self, model_path, session_factory):
            path = Path(str(model_path))
            if not path.is_file():
                raise FileNotFoundError(f'Model not found: {path}')
            self.model_path = str(path)
            self._factory = session_factory
            self.model = OnnxNet(path, session_factory)
            self.provider = self.model.provider
            self.note = self.model.note
            self.n_cond = 6
            self.has_star_controls = True

        def run(self, data, params=None, progress=None, cancel=None):
            p = dict(params or {})
            data = np.ascontiguousarray(data, np.float32)
            fb = is_fb(self.model_path)
            correct_only = bool(p.get('correct_only', False))
            strength = float(p.get('strength', 0.5)); star = float(p.get('star', 0.0)); halo = float(p.get('halo', 0.0))
            sharpening = float(p.get('sharpening', SHARP_DEFAULT)) if fb else None
            if correct_only:
                strength, star, halo = 0.0, 0.0, 0.0
                if fb:
                    sharpening = 0.0
            fwhm = p.get('fwhm')
            fd = is_finedetail(self.model_path); sm = None
            if fd and p.get('protect_stars') and sharpening:
                sp = p.get('star_model_path')
                if not sp or not Path(str(sp)).is_file():
                    raise FileNotFoundError(f'Protect star shapes needs the Umbra Deblur v1.0 model ({sp})')
                sm = OnnxNet(sp, self._factory)
            if cancel is not None:
                inner = progress
                def progress(n, t, _inner=inner):
                    if cancel():
                        raise RuntimeError('Cancelled')
                    if _inner:
                        _inner(n, t)
            out, info = deconvolve_array(self.model, data, strength, fwhm, progress, star, halo, sharpening, p.get('drizzled'),
                                         finedetail=fd, star_model=sm)
            g = info.get('fwhm_global')
            info['fwhm'] = None if g is None else float(g)
            if 'notes' not in info:
                info['notes'] = [info['psf_note']] if info.get('psf_note') else []
            return out, info


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
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


ENGINES['deblur'] = _engine_deblur()

# ============================================================================================ tools
def _tool_deblur():
    # ---- Umbra Deblur: tool layer (panel, CLI flags, labels) ----
    SHARP_DEFAULT = 3.0
    STRENGTH_TIP = 'How far to sharpen toward the sampling limit; always limited by the noise. 0 = Aberration Removal Only.'
    STAR_TIP = 'Shrinks stars toward the sampling limit (noise-limited). Only this slider changes star size.'
    HALO_TIP = 'Pulls star halos in; their light goes back into the star core (brightness kept).'
    CORRECT_TIP = 'Fixes star shapes (aberrations) without sharpening. Switches the other sliders off.'
    SHARP_TIP = ('Number of sharpening passes (PSF re-measured each time); 2.5 = 2 passes + a half-strength pass. 0 = Aberration Removal Only. '
                 'Umbra Deblur v1.0: every step is sharper than the last; 8 is the strongest; cleaner dark dust lanes. '
                 'v1.1 (experimental): sharper fine strands, less beading; sky mask on automatically; may keep slight coma tails on bright '
                 'corner stars unless Protect star shapes is ticked. v1.0 stays the default. Images differ, so judge by eye.')
    PROTECT_TIP = ('Fine-detail model (v1.1) only: stars and their coma tails come from Umbra Deblur v1.0 (best star shapes), all other '
                   'detail from v1.1. About 2x slower.')
    PSF_TIP = 'Manual star diameter (FWHM) in pixels.'
    STAR_MODEL_FILES = ('umbra_deblur_v1.0.onnx', 'umbra_deconvolution_1.0.onnx')


    def _is_drizzled(text):
        return 'drizzle' in str(text or '').lower()


    class DeblurPanel:
        def __init__(self, ui, layout, host):
            self.ui, self.host = ui, host
            self.fb = True
            self.finedetail = False
            self.correct = ui.check(layout, 'Aberration Removal Only', CORRECT_TIP)
            self.strength = ui.slider(layout, 'Strength', 0, 100, 50, lambda v: f'{v}%', STRENGTH_TIP)
            self.sharp = ui.slider(layout, 'Sharpening', 0, 32, int(SHARP_DEFAULT * 4), lambda v: f'{v / 4:.2f}', SHARP_TIP)   # 0..8 in 0.25 steps
            self.protect = ui.check(layout, 'Protect star shapes (slower)', PROTECT_TIP)
            self.auto = ui.check(layout, 'Automatic PSF', 'Measures the star size from the image. Switch off to set the PSF Diameter by hand.', True)
            self.psf = ui.slider(layout, 'PSF Diameter (px)', 15, 100, 30, lambda v: f'{v / 10:.1f}', PSF_TIP)
            self.star = ui.slider(layout, 'Sharpen Stars', 0, 100, 0, lambda v: f'{v}%', STAR_TIP)
            self.halo = ui.slider(layout, 'Halo Reduction', 0, 100, 0, lambda v: f'{v}%', HALO_TIP)
            self.no_stars = None                 # set from the background star check when an image is loaded
            self.auto_pref = True                # the user's own Automatic PSF choice (kept while the box is greyed out)
            self.auto_tip = self.auto.toolTip()
            self.correct.toggled.connect(lambda _=False: self.update())
            self.auto.toggled.connect(lambda c: (setattr(self, 'auto_pref', bool(c)) if self.auto.isEnabled() else None, self.update()))
            self.update()

        # background check when the window loads an image (MainWindow calls image_check in a worker thread, then image_checked)
        def image_check(self, data):
            x = np.asarray(data, np.float32)
            if x.ndim == 3 and x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
                x = np.moveaxis(x, -1, 0)
            return ENGINES['deblur'].psf_check(x)

        def image_checked(self, res):
            self.no_stars = not res['ok']
            if self.no_stars:
                self.auto.blockSignals(True); self.auto.setChecked(False); self.auto.blockSignals(False)
                self.auto.setEnabled(False)
                self.auto.setToolTip(f"No stars found ({res['density']:.0f} isolated stars per megapixel): Automatic PSF needs stars with the current "
                                     'models. Set the PSF Diameter by hand (measure it on the version of the image with stars).')
            else:
                self.auto.setEnabled(True)
                self.auto.setToolTip(self.auto_tip)
                self.auto.blockSignals(True); self.auto.setChecked(self.auto_pref); self.auto.blockSignals(False)
            self.update()

        def update(self):
            co = self.correct.isChecked()
            self.ui.enable(self.strength, not co); self.ui.enable(self.sharp, not co)
            self.ui.show(self.strength, not self.fb); self.ui.show(self.sharp, self.fb)
            self.protect.setVisible(self.finedetail)
            self.ui.enable(self.psf, not self.auto.isChecked())
            self.ui.enable(self.star, not co); self.ui.enable(self.halo, not co)

        def model_changed(self, path):
            eng = ENGINES['deblur']
            self.fb = eng.is_fb(path) if path else True
            self.finedetail = bool(path) and eng.is_finedetail(path)
            self.update()

        def star_model(self):
            d = self.ui.model_dir()
            for f in STAR_MODEL_FILES:
                if (d / f).is_file():
                    return str(d / f)
            return None

        def params(self):
            p = {'correct_only': self.correct.isChecked(), 'fwhm': None if self.auto.isChecked() else self.psf.value() / 10.0,
                 'star': self.star.value() / 100.0, 'halo': self.halo.value() / 100.0}
            if self.fb:
                p['sharpening'] = self.sharp.value() / 4.0
            else:
                p['strength'] = self.strength.value() / 100.0
            if self.protect.isVisible() and self.protect.isChecked():
                p['protect_stars'] = True
                p['star_model_path'] = self.star_model()
            return p

        def settings(self):
            return dict(correct=self.correct.isChecked(), strength=self.strength.value(), sharp=self.sharp.value(), protect=self.protect.isChecked(),
                        auto=self.auto_pref, psf=self.psf.value(), star=self.star.value(), halo=self.halo.value())

        def set_settings(self, d):
            for k, w in (('strength', self.strength), ('sharp', self.sharp), ('psf', self.psf), ('star', self.star), ('halo', self.halo)):
                if isinstance(d.get(k), int):
                    w.setValue(d[k])
            for k, w in (('correct', self.correct), ('protect', self.protect), ('auto', self.auto)):
                if isinstance(d.get(k), bool):
                    w.setChecked(d[k])
            if isinstance(d.get('auto'), bool):
                self.auto_pref = d['auto']
            self.update()


    class DeblurSpec(ToolSpec):
        def build_panel(self, ui, layout, host):
            return DeblurPanel(ui, layout, host)

        def add_cli(self, ap):
            ap.add_argument('--strength', type=float, default=0.5, help='deblur: 0..1 (default 0.5; models without Sharpening only)')
            ap.add_argument('--sharpening', type=float, help='deblur: passes 0..8 in steps of 0.25 (default 3; 2.5 = 2 passes + half pass)')
            ap.add_argument('--stars', type=float, default=0.0, help='deblur: Sharpen Stars 0..1 (default 0)')
            ap.add_argument('--halo', type=float, default=0.0, help='deblur: Halo Reduction 0..1 (default 0)')
            ap.add_argument('--correct-only', action='store_true', help='deblur: Aberration Removal Only (force strength, sharpening, stars and halo to 0)')
            ap.add_argument('--fwhm', type=float, help='deblur: manual PSF FWHM in px (default: automatic)')
            ap.add_argument('--protect-stars', action='store_true', help='deblur: v1.1 only: star shapes from v1.0 (~2x slower)')

        def check_cli(self, a, ap):
            if not 0 <= a.strength <= 1:
                ap.error('--strength must lie in [0, 1]')
            if a.fwhm is not None and not 0.5 <= a.fwhm <= 20:
                ap.error('--fwhm must be between 0.5 and 20 px')
            if a.sharpening is not None and not 0 <= a.sharpening <= 8:
                ap.error('--sharpening must lie in 0..8')
            if not 0 <= a.stars <= 1 or not 0 <= a.halo <= 1:
                ap.error('--stars and --halo must lie in [0, 1]')

        def params_from_args(self, a):
            p = {'correct_only': bool(a.correct_only), 'fwhm': a.fwhm, 'star': a.stars, 'halo': a.halo, 'strength': a.strength}
            if a.sharpening is not None:
                p['sharpening'] = a.sharpening
            if a.protect_stars:
                p['protect_stars'] = True
                d = model_dir(self)[0]
                p['star_model_path'] = next((str(d / f) for f in STAR_MODEL_FILES if (d / f).is_file()), None)
            return p

        def image_params(self, header_text):
            return {'drizzled': _is_drizzled(header_text)}      # drizzled stacks have correlated noise: the engine raises its noise estimate

        def label(self, p):
            head = f"sharpening {float(p['sharpening']):.2f}" if 'sharpening' in p else f"strength {float(p.get('strength', 0.5)):.2f}"
            if p.get('correct_only'):
                head = 'correct only'
            psf = 'auto PSF' if p.get('fwhm') is None else f"PSF {float(p['fwhm']):.1f} px"
            return f"{head} stars {float(p.get('star', 0)):.2f} halo {float(p.get('halo', 0)):.2f} {psf}" + (' protect-stars' if p.get('protect_stars') else '')


    SPEC = DeblurSpec(
        key='deblur', name='Umbra Deblur', script='Umbra_Deblur', repo=f'{GITHUB_USER}/umbra-deblur', legacy_repos=(f'{GITHUB_USER}/umbra-deconvolve',),
        prefix='UMBRA_DEBLUR_', folder='Umbra Deblur', fallback_folder='Umbra Deconvolve',
        labels={'umbra_deblur_v1.0': 'Umbra Deblur v1.0 (default)', 'umbra_deblur_v1.1_experimental': 'Umbra Deblur v1.1 (experimental)',
                'umbra_deconvolution_1.0': 'Umbra Deblur v1.0 (default)', 'umbra_deconvolution_1.1a_experimental': 'Umbra Deblur v1.1 (experimental)'},
        default_stems=('umbra_deblur_v1.0', 'umbra_deconvolution_1.0'),
        notes=['AI deblur (deconvolution) for linear images.'])
    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})


TOOLS['deblur'] = _tool_deblur().SPEC


if __name__ == '__main__':
    raise SystemExit(main([TOOLS[k] for k in ('deblur',)], combined=False, doc=__doc__))
