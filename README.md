# Umbra Deblur (BETA)

Umbra Deblur is an AI deblur (deconvolution) tool for [Siril](https://siril.org), aimed at small-scope images (Seestar, Dwarf and similar) but usable on any linear image. It is a Siril Python script with an AI model. It runs locally on your computer and is free. It used to be called Umbra Deconvolve.

**Status: Beta (v0.2.0).** Expect rough edges.

![Umbra Deblur panel](docs/screenshots/Umbra_Deblur_panel.png)

![Umbra Deblur panel](docs/screenshots/Umbra_Deblur_panel_updnote.png)

## Download and install

1. Download the single file [`Umbra_Deblur.py`](https://raw.githubusercontent.com/Kishanhpatel/umbra-deblur/main/Umbra_Deblur.py) (also attached to the [latest release](https://github.com/Kishanhpatel/umbra-deblur/releases/latest)). Save it in any folder you like. Siril has no standard scripts folder name.
2. In Siril open **Preferences > Scripts**, add that folder, refresh the list, then run **Umbra Deblur** from the Scripts menu.
3. First run: Siril installs the Python packages the script needs (numpy, onnxruntime, PyQt6, scipy, astropy). This can take a few minutes.
4. If no model is installed, press **Download model** (Umbra Deblur v1.0, about 199 MB). The file is checked with SHA-256 and saved in a folder named `Umbra Deblur` next to the script.

**Manual install.** Download the `.onnx` file from the [release page](https://github.com/Kishanhpatel/umbra-deblur/releases/latest) and put it in the `Umbra Deblur` folder next to the script. To use another folder, set `UMBRA_DEBLUR_MODEL_DIR` to it. `UMBRA_DEBLUR_MODEL` can point at one specific model file.

**Updates.** **Check for updates** compares your installed models and the script version with the published list (`models/models.json`). A new or changed model downloads only when you press the button. A newer script is shown as a link to the release page; the script never replaces itself. Nothing is sent about you or your computer (only a generic client name). With no network the installed models keep working. Set `UMBRA_DEBLUR_OFFLINE=1` (or `UMBRA_OFFLINE=1`) to turn the lookup off.

**Coming from Umbra Deconvolve 0.0.14?** This is the same project under a new name. Models in the old `Umbra Deconvolve` folder are used automatically (when the `Umbra Deblur` folder has none), so nothing downloads twice. The old repository link redirects here, and the old script keeps working with its old models. The environment variables changed from `UMBRA_DECONV_*` to `UMBRA_DEBLUR_*`.

## What it does

It measures the star shape (PSF) of your image, then runs the model for a number of passes to sharpen it. Controls:

- **Correct Only**: fixes star shapes (aberrations) without sharpening. Switches the sharpening, star and halo controls off.
- **Sharpening** (0 to 8, steps of 0.25, default 3): the number of sharpening passes. The star shape is measured again each pass. 2.5 means 2 passes plus a half-strength pass. 0 is the same as Correct Only. (Older models without this setting show **Strength** 0 to 100% instead.)
- **Protect star shapes (slower)**: only shown for v1.1 (experimental). Stars come from v1.0, which gives the best star shapes, and all other detail from v1.1. About twice as slow.
- **Automatic PSF** and **PSF Diameter (px)** (1.5 to 10): the star size is measured from the image by default. Switch off to set it by hand.
- **Sharpen Stars** (0 to 100%): shrinks stars toward the sampling limit. Only this slider changes star size. Default 0.
- **Halo Reduction** (0 to 100%): pulls star halos in and puts their light back into the star core. Default 0.

If the FITS header mentions drizzle, the noise estimate is raised, because drizzled stacks have correlated noise. If the image looks stretched, the status line warns you; the tool expects linear data.


**Preview.** The **Preview** button opens a separate, resizable window. It shows a region of the image before and after (same stretch for both): your current Siril selection if it is at least 64 px, otherwise a centred 768 x 768 crop (the whole image if smaller). Small selections are processed with some surrounding image for context, then cropped back. Hold down **Show original** (or the Space key) to see the original; release it to see the result again. The star shape is measured on that region only. It never changes the image in Siril.

**Apply.** **Apply** replaces the image in Siril with the result. Use Siril's **Undo** to go back. Siril's scripting interface cannot write a FITS HISTORY line, so the settings used are written to the Siril log (and into any extra FITS file the tool saves).

**Command line.** `python Umbra_Deblur.py --image IN.fits --out OUT.fits` runs the tool without a window. Flags: `--sharpening 0..8`, `--stars`, `--halo`, `--correct-only`, `--fwhm PX`, `--protect-stars`, `--model PATH`. `--check-updates` and `--download [MODEL_ID|all]` work from the command line too, and `--help` lists everything.

## Models

Models are published as GitHub release assets and are not stored in this repository. See `models/models.json`.

| Name | File | Size | Notes |
|---|---|---|---|
| Umbra Deblur v1.0 (default) | `umbra_deblur_v1.0.onnx` | 199 MB | Clean, stable medium-scale sharpening. Best star shapes. |
| Umbra Deblur v1.1 (experimental) | `umbra_deblur_v1.1_experimental.onnx` | 199 MB | Joins fine 1 to 4 pixel strands better. Turns on a sky mask automatically. May change. |

Old names (Umbra Deconvolve 0.0.14), same files:

| Old name | Old file | New name | New file |
|---|---|---|---|
| Umbra Deconvolution 1.0 | `umbra_deconvolution_1.0.onnx` | Umbra Deblur v1.0 | `umbra_deblur_v1.0.onnx` |
| Umbra Deconvolution 1.1a experimental | `umbra_deconvolution_1.1a_experimental.onnx` | Umbra Deblur v1.1 (experimental) | `umbra_deblur_v1.1_experimental.onnx` |

The old files are byte-for-byte the same models. If you already have them they keep working.

## Strengths

- Sharpens medium-scale detail cleanly, and stays stable up to Sharpening 8 on most targets (tested on a 22-image test bank).
- Dark dust lanes get narrower instead of turning into blobs.
- Stars stay round. Sky grain is checked and not increased.
- Handles starless images, drizzled data and saturated stars.
- v1.1 joins fine strands better (for example the Arp 273 arcs, Crescent Nebula strands and the inner M51), and has an optional "Protect star shapes" box.
- GPU acceleration with automatic CPU fallback.

## Weaknesses

- Very fine (1 to 4 pixel) detail is the weak spot. It is most visible inside bright cores (the inner M51 bulge, the Centaurus A core), where filaments stay smooth or break into beads.
- At Sharpening 8 some galaxies (M101, NGC 6946, Arp 273) can show mild pits or beading. Back off if you see it.
- v1.1 can leave coma-like tails on bright corner stars unless "Protect star shapes" is on. That costs roughly 1.5 to 2 times the run time.
- Star peaks can go above 1.0 at high settings, so save results as 32-bit float.
- Linear data only. Stretched images are not what it was made for.
- Tested mainly on the author's own data and one Mac.

## GPU acceleration

The GPU is used by default whenever one can be used, and the CPU only as a fallback. Providers are tried in this order: CUDA (NVIDIA), DirectML (Windows), ROCm and MIGraphX (AMD on Linux), OpenVINO (Intel), Core ML (Mac), then CPU. Each one is checked against the CPU result on a small probe tile before it is used; if it fails or the result differs, the script falls back to the CPU. The status line says which one ran ("Ran on ..."). Set `UMBRA_DEBLUR_CPU=1` to force the CPU.

On first run the script installs the onnxruntime package that gives your system a GPU:

- **Mac:** `onnxruntime` (Core ML, Apple GPU, built in).
- **Windows:** `onnxruntime-directml` (any DirectX 12 GPU: NVIDIA, AMD or Intel; no extra downloads).
- **Linux with an NVIDIA GPU:** `onnxruntime-gpu` (CUDA). If the system has no CUDA 12 / cuDNN 9 libraries, NVIDIA's own pip packages of them are installed too (a one-time download of about 2 to 3 GB).
- **Linux with AMD or Intel graphics:** there is no GPU build that installs automatically, so these run on the CPU. If you install `onnxruntime-rocm`, `onnxruntime-migraphx` or `onnxruntime-openvino` yourself in Siril's Python, the script uses it.

## Testing

Be aware of what has and has not been tried. This is a beta.

Tested:
- macOS 26 on an Apple M3, with Core ML and with the CPU.
- Standalone (command line and window test) and inside Siril 1.4.5 run headless (`siril-cli` with `pyscript`): read the image, process it, write it back with an Undo state, save.
- The window was tested offscreen only.

NOT tested:
- Windows and Linux.
- CUDA, DirectML and ROCm (the code is written; please report what you see).
- Intel Macs.
- The interactive window inside the Siril GUI.

Parity with the earlier versions: on real 768 px crops (three targets, both models, Sharpening 0, 3 and 6, Protect star shapes on, a mono crop, a 300 x 250 crop and a crop with a NaN hole) the output is identical to the previous Umbra Deconvolve 0.0.14 (maximum difference 0). That version matched the original PyTorch version to 0.01 to 0.07 noise sigma. Core ML and CPU differ by at most 0.02 sigma.
Small images: the model uses fixed 512 px tiles. A 300 x 250 crop matched the reference exactly, but only one small case was tried.

## Layout

- `Umbra_Deblur.py`: the script (v0.2.0, ONNX Runtime, with model download and update check)
- `models/models.json`: model manifest (name, version, download URL, SHA-256, size)
- `docs/screenshots/`: panel screenshots
- `LICENSE`, `LICENSE-MODELS.md`: licenses

## Contact

Bugs or questions: umbraastrolabs@gmail.com, or open an issue on GitHub.

## License

- Script: GPL-3.0-only (see `LICENSE`).
- Models: CC BY-NC-SA 4.0, non-commercial (see `LICENSE-MODELS.md`).
