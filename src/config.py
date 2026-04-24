import json
from pathlib import Path
from dataclasses import dataclass, field

from features import FeatureExtractorConfig
from light_curves import PreprocessingConfig


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            features=FeatureExtractorConfig(**raw.get("features", {})),
        )

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({
                "preprocessing": vars(self.preprocessing),
                "features": vars(self.features),
            }, f, indent=2)