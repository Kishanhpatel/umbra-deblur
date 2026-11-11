# Umbra Deconvolve

AI deconvolution (sharpening) script for [Siril](https://siril.org), built for Seestar, Dwarf and other small-scope images.

> **Status: work in progress, not released.** The script and models are being converted to ONNX Runtime. Nothing here is final.

## Install (draft)

1. Download `Umbra_Deconvolve.py` into your Siril scripts folder.
2. Download the model listed in [`models/models.json`](models/models.json) from the Releases page. Models are release assets and are not stored in this repository.
3. In Siril, add the scripts folder under Preferences > Scripts, then run **Umbra Deconvolve**.

## Models

- **Umbra Deconvolution 1.0**: the default.
- **Umbra Deconvolution 1.1a experimental**: optional, may change.

(Whether both ship in the first release is not decided yet.)

## Layout

- `Umbra_Deconvolve.py`: the script (placeholder until the ONNX version lands)
- `models/models.json`: model manifest (name, version, download URL, SHA-256)
- `docs/`: documentation; `docs/screenshots/` for images
- `LICENSE`: placeholder, license not yet chosen

## License

Not yet chosen. See `LICENSE`.
