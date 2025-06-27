import transformers
import numpy as np
from pathlib import Path
import librosa

import json


class VoiceEmbedderProcessor(
    transformers.SequenceFeatureExtractor, transformers.ProcessorMixin
):
    model_input_names = ["input_features", "attention_mask"]
    _processor_class = "VoiceEmbedderProcessor"

    def __init__(
        self,
        sequence_length: int = 1000,
        whisper_model: str = "openai/whisper-base",
        feature_extractor=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            feature_size=80,
            sampling_rate=16000,
            padding_value=0.0,
            processor_class=self._processor_class,
        )
        self.sequence_length = sequence_length
        self.feature_extractor = (
            feature_extractor
            or transformers.WhisperFeatureExtractor.from_pretrained(whisper_model)
        )

    def __call__(
        self,
        audio: list[np.ndarray] | np.ndarray,
        sampling_rate=None,
        sequence_length=None,
        **kwargs,
    ):
        target_sr = 16000
        kwargs["return_attention_mask"] = kwargs.pop("return_attention_mask", True)

        is_single_audio = not isinstance(audio, (list, tuple))
        if is_single_audio:
            audio = [audio]
            sampling_rate = (
                [sampling_rate] if sampling_rate is not None else [target_sr]
            )

        if sampling_rate is None:
            sampling_rate = [target_sr] * len(audio)
        elif isinstance(sampling_rate, (int, float)):
            sampling_rate = [sampling_rate] * len(audio)
        elif len(sampling_rate) != len(audio):
            raise ValueError(
                "Number of sampling rates must match number of audio inputs"
            )

        resampled_audio = []
        for aud, sr in zip(audio, sampling_rate):
            if not isinstance(aud, np.ndarray):
                aud = np.array(aud, dtype=np.float32)
            if sr != target_sr:
                aud = librosa.resample(aud, orig_sr=sr, target_sr=target_sr)
            resampled_audio.append(aud)

        features = self.feature_extractor(
            resampled_audio, sampling_rate=target_sr, **kwargs
        )

        sequence_length = sequence_length or self.sequence_length

        features["input_features"] = features["input_features"][:, :, :sequence_length]
        if "attention_mask" in features:
            features["attention_mask"] = features["attention_mask"][:, :sequence_length]

        return features

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        sequence_length = kwargs.pop("sequence_length", 1000)  # Default sequence length
        return cls(
            feature_extractor=feature_extractor,
            sequence_length=sequence_length,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
            f"feature_size={self.feature_size}, sampling_rate={self.sampling_rate}, "
            f"padding_value={self.padding_value})"
        )

    def save_pretrained(self, save_directory, **kwargs):
        config = self.feature_extractor.to_dict()
        config["sequence_length"] = self.sequence_length
        config["processor_class"] = self._processor_class
        config.pop("_processor_class")  # Inherited from self.feature_extractor

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        with (save_directory / "preprocessor_config.json").open("w") as f:
            json.dump(config, f, indent=2)
