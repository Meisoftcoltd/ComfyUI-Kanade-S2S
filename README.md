# ComfyUI-Kanade-S2S

ComfyUI custom nodes for the [Kanade: A Simple Disentangled Tokenizer for Spoken Language Modeling](https://github.com/frothywater/kanade-tokenizer).

These nodes allow you to encode and decode speech using the Kanade speech tokenizer. It supports the latest `kanade-25hz-clean` model with HiFT vocoder.

## Nodes

* **Kanade Model Loader**: Downloads and loads the Kanade model and Vocoder weights directly into ComfyUI's standard `models/kanade` folder.
* **Kanade Encoder**: Takes standard ComfyUI `AUDIO` as input, and outputs disentangled `KANADE_TOKENS` and `KANADE_EMBEDDING`.
* **Kanade Decoder**: Takes `KANADE_TOKENS` and `KANADE_EMBEDDING` and reconstructs them back into standard ComfyUI `AUDIO` output.

## Installation

To install, you can either:

1. Clone this repository into your `ComfyUI/custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone <this_repo_url>
cd ComfyUI-Kanade-S2S
pip install -r requirements.txt
```
2. Or install via the ComfyUI Manager.

### Windows / `flash-attn` Warning

The base Kanade model utilizes `flash-attn` for efficient local window attention.
Windows users often have trouble compiling this library. We recommend installing it using the following command in your ComfyUI python environment:

```bash
uv pip install flash-attn --no-build-isolation
```
*(Ensure `ninja` is installed on your system or the build will be very slow)*

If `flash-attn` is not available, the model will fall back to regular PyTorch SDPA implementation, which may not guarantee the same quality reported in the original paper.
