# Umbra Deconvolve

AI deconvolution (sharpening) for [Siril](https://siril.org), aimed at small-scope images (Seestar, Dwarf and similar) but usable on any linear image. The goal is a free alternative to commercial AI deconvolution tools. It does not yet match them on the very finest detail (see Weaknesses).

> **Status: preview (v0.0.13).** Expect rough edges. Not yet tested inside Siril on Windows or Linux.

## What it does

A Siril Python script runs a neural network on your image to sharpen it. It measures the star PSF of the image first, then applies a number of passes ("Sharpening" setting). It uses ONNX Runtime, so no PyTorch is needed, and it uses the GPU when it can, with a CPU fallback.

## Models

- **Umbra Deconvolution 1.0** (default): clean, stable medium-scale sharpening.
- **Umbra Deconvolution 1.1a experimental**: better at joining fine 1 to 4 pixel strands. May change.

Models are published as GitHub Release assets and are not stored in this repository. See `models/models.json`.

## Strengths

- Sharpens medium-scale detail cleanly, and stays stable up to Sharpening 8 on most targets (tested on a 22-image test bank).
- Dark dust lanes get narrower instead of turning into blobs.
- Stars stay round. Sky grain is checked and not increased.
- Handles starless images, drizzled data and saturated stars.
- 1.1a joins fine strands better (for example the Arp 273 arcs, Crescent strands and the inner M51), and turns on a sky mask automatically. An optional "Protect star shapes (slower)" box is available.
- GPU acceleration with automatic CPU fallback. Each GPU provider is checked against the CPU result on a small test tile before it is used.

## Weaknesses

- Very fine (1 to 4 pixel) detail still falls short of BlurXTerminator, most visibly inside bright cores (inner M51 bulge, Centaurus A core), where filaments stay smooth or break into beads.
- At Sharpening 8 some galaxies (M101, NGC 6946, Arp 273) can show mild pits or beading. Back off if you see it.
- 1.1a can leave coma-like tails on bright corner stars unless "Protect star shapes" is on, which costs roughly 1.5 to 2 times the run time.
- Star peaks can go above 1.0 at high settings, so save results as 32-bit float.
- Tested mainly on the author's own data and one Mac.
- **GPU support is only tested on Mac** (Core ML and CPU). The CUDA, DirectML and ROCm paths are written but untested. Please report what you see.
- Not tested inside Siril on Windows or Linux yet.

## GPU acceleration

Providers are tried in this order: CUDA, DirectML (Windows), ROCm (AMD on Linux), Core ML (Mac), then CPU. The status line says which one ran ("Ran on ..."). Set `UMBRA_DECONV_CPU=1` to force the CPU. On first run the script installs `onnxruntime-directml` (Windows), `onnxruntime-gpu` (Linux with an NVIDIA GPU) or plain `onnxruntime` (everything else).

## Install (draft)

1. Download `Umbra_Deconvolve.py` into your Siril scripts folder.
2. Download the model files from the Releases page into an `Umbra Deconvolve` folder next to the script.
3. In Siril, add the scripts folder under Preferences > Scripts, then run **Umbra Deconvolve**.

## Layout

- `Umbra_Deconvolve.py`: the script (v0.0.13, ONNX Runtime)
- `models/models.json`: model manifest (name, version, download URL, SHA-256)
- `docs/`: documentation, with `docs/screenshots/` for images

## License

- Script: GPL-3.0-only (see `LICENSE`).
- Models: CC BY-NC-SA 4.0, non-commercial (see `LICENSE-MODELS.md`).

These match the licenses GraXpert uses for its code and its AI models.
