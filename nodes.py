import os
import torch
import torchaudio
import folder_paths
from huggingface_hub import snapshot_download

# Define and register the "kanade" model folder path
kanade_models_dir = os.path.join(folder_paths.models_dir, "kanade")
os.makedirs(kanade_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("kanade", kanade_models_dir)

try:
    from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode
except ImportError:
    print("Kanade-tokenizer package is not installed. Please install it using 'pip install git+https://github.com/frothywater/kanade-tokenizer'.")


class KanadeModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["kanade-12.5hz", "kanade-25hz", "kanade-25hz-clean"], {"default": "kanade-25hz-clean"}),
            }
        }

    RETURN_TYPES = ("KANADE_MODEL", "KANADE_VOCODER")
    RETURN_NAMES = ("kanade_model", "kanade_vocoder")
    FUNCTION = "load_model"
    CATEGORY = "Audio/Kanade"

    def load_model(self, model_name):
        repo_map = {
            "kanade-12.5hz": "frothywater/kanade-12.5hz",
            "kanade-25hz": "frothywater/kanade-25hz",
            "kanade-25hz-clean": "frothywater/kanade-25hz-clean"
        }
        repo_id = repo_map.get(model_name)

        # Get the registered folder paths for kanade models
        download_dir = folder_paths.get_folder_paths("kanade")[0]
        model_path = os.path.join(download_dir, model_name)

        print(f"Checking/Downloading weights for {repo_id} to {model_path}...")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            allow_patterns=["*.safetensors", "*.json", "*.yaml"]
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model
        model = KanadeModel.from_pretrained(snapshot_path)
        model = model.eval().to(device)

        # Load vocoder
        vocoder = load_vocoder(model.config.vocoder_name).to(device)
        vocoder.eval()

        return (model, vocoder)


class KanadeEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kanade_model": ("KANADE_MODEL",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("KANADE_TOKENS", "KANADE_EMBEDDING")
    RETURN_NAMES = ("content_tokens", "global_embedding")
    FUNCTION = "encode"
    CATEGORY = "Audio/Kanade"

    def encode(self, kanade_model, audio):
        device = next(kanade_model.parameters()).device

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Squeeze batch dimension if present and make sure it has shape (1, samples)
        if waveform.dim() == 3:
            # shape (batch, channels, samples) -> typically batch is 1 for comfyui audio
            waveform = waveform[0]

        if waveform.dim() > 2:
            waveform = waveform.squeeze(0) # (channels, samples)

        # If stereo, mean to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = getattr(kanade_model.config, 'sample_rate', 24000)

        if sample_rate != target_sr:
            print(f"Resampling audio from {sample_rate} to {target_sr}")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(device)
            waveform = resampler(waveform.to(device))
        else:
            waveform = waveform.to(device)

        # Ensure shape is (samples,)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        with torch.no_grad():
            features = kanade_model.encode(waveform)

        return (features.content_token_indices, features.global_embedding)


class KanadeDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kanade_model": ("KANADE_MODEL",),
                "kanade_vocoder": ("KANADE_VOCODER",),
                "content_tokens": ("KANADE_TOKENS",),
                "global_embedding": ("KANADE_EMBEDDING",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "Audio/Kanade"

    def decode(self, kanade_model, kanade_vocoder, content_tokens, global_embedding):
        device = next(kanade_model.parameters()).device

        content_tokens = content_tokens.to(device)
        global_embedding = global_embedding.to(device)

        with torch.no_grad():
            # Synthesize audio from extracted features
            mel_spectrogram = kanade_model.decode(
                content_token_indices=content_tokens,
                global_embedding=global_embedding,
            )

            # vocode expects (1, n_mels, T)
            # mel_spectrogram comes out as (n_mels, T), so unsqueeze
            if mel_spectrogram.dim() == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)

            resynthesized_waveform = vocode(kanade_vocoder, mel_spectrogram) # (1, samples)

        # Format back into ComfyUI Audio format: (1, channels, samples)
        # resynthesized_waveform is (1, samples)
        waveform_out = resynthesized_waveform.unsqueeze(0).cpu() # (1, 1, samples)

        sample_rate = getattr(kanade_model.config, 'sample_rate', 24000)

        audio_out = {
            "waveform": waveform_out,
            "sample_rate": sample_rate
        }

        return (audio_out,)
