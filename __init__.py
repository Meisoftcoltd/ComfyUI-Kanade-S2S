import os
import sys

# Add current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .nodes import KanadeModelLoader, KanadeEncoder, KanadeDecoder

NODE_CLASS_MAPPINGS = {
    "KanadeModelLoader": KanadeModelLoader,
    "KanadeEncoder": KanadeEncoder,
    "KanadeDecoder": KanadeDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KanadeModelLoader": "Kanade Model Loader",
    "KanadeEncoder": "Kanade Encoder",
    "KanadeDecoder": "Kanade Decoder",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
