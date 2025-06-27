from .model import VoiceEmbedder, VoiceEmbedderConfig
from .dataset import VoiceEmbedderDataset, collate_fn
from .preprocessor import VoiceEmbedderProcessor
from .inference_utils import infer


import transformers


transformers.AutoConfig.register("voice_embedder", VoiceEmbedderConfig)
transformers.AutoModel.register(VoiceEmbedderConfig, VoiceEmbedder)
transformers.AutoProcessor.register("VoiceEmbedderProcessor", VoiceEmbedderProcessor)

__all__ = [
    "VoiceEmbedderProcessor",
    "VoiceEmbedder",
    "VoiceEmbedderConfig",
    "VoiceEmbedderDataset",
    "collate_fn",
]
