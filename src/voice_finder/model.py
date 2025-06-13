import transformers
import torch


class VoiceEmbedderConfig(transformers.WhisperConfig):
    model_type = "voice_embedder"

    def __init__(
        self,
        sequence_length: int = 1000,
        projector_dim: int = 256,
        whisper_model: str = "openai/whisper-base",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.projector_dim = projector_dim
        self.whisper_model = whisper_model


class VoiceEmbedder(transformers.PreTrainedModel):
    config_class = VoiceEmbedderConfig

    def __init__(self, config):
        super().__init__(config)

        self.encoder = self._load_whisper_encoder(config)

        self.projector_head = torch.nn.Sequential(
            torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256)
        )

    def _load_whisper_encoder(self, config):
        model_state_dict = transformers.WhisperModel.from_pretrained(
            config.whisper_model
        ).state_dict()

        # Shorten embedding length to desired sequence_length
        model_state_dict["encoder.embed_positions.weight"] = model_state_dict[
            "encoder.embed_positions.weight"
        ][0 : config.sequence_length // 2, :]

        whisper_config = transformers.WhisperConfig.from_pretrained(
            config.whisper_model, max_source_positions=config.sequence_length // 2
        )

        model = transformers.WhisperModel(whisper_config)

        model.load_state_dict(model_state_dict)

        return model.encoder

    def forward(self, input_features, attention_mask=None):
        encoder_out = self.encoder(
            input_features=input_features, attention_mask=attention_mask
        )

        hidden_state = encoder_out.last_hidden_state.mean(dim=1)

        embedding = self.projector_head(hidden_state)

        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding
