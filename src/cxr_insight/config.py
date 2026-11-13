"""Typed, validated experiment configuration loaded from TOML or JSON.

Dependency-free (stdlib only). Relative paths are resolved against the config
file's directory so a config behaves the same regardless of the working directory.
"""
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    image_root: Path
    output_dir: Path
    seed: int = 42
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self):
        _check_int("data.seed", self.seed, minimum=0)
        for name in ("validation", "test"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"data.{name} must be a number")
        # Same rule as data.split_patients, so a config cannot describe an invalid split.
        if not (0 < self.validation < 1 and 0 < self.test < 1
                and self.validation + self.test < 1):
            raise ValueError("data.validation/test must be positive and sum to < 1")


@dataclass(frozen=True)
class ModelConfig:
    base_model_id: str
    revision: str | None = None
    adapter_path: Path | None = None
    image_encoder: str = "efficientnet_b0"

    def __post_init__(self):
        if not isinstance(self.base_model_id, str) or not self.base_model_id.strip():
            raise ValueError("model.base_model_id must be a non-empty string")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("model.revision must be a non-empty string when set")
        if not isinstance(self.image_encoder, str) or not self.image_encoder.strip():
            raise ValueError("model.image_encoder must be a non-empty string")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    seed: int = 42

    def __post_init__(self):
        lr = self.learning_rate
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr < 1:
            raise ValueError("training.learning_rate must be in (0, 1)")
        for name in ("epochs", "batch_size", "gradient_accumulation_steps",
                     "max_seq_length", "lora_r", "lora_alpha"):
            _check_int(f"training.{name}", getattr(self, name), minimum=1)
        _check_int("training.seed", self.seed, minimum=0)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

    def to_dict(self):
        """JSON-serialisable view, for saving alongside results."""
        return json.loads(json.dumps(asdict(self), default=str))


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _build(cls, section, values, base_dir):
    if not isinstance(values, dict):
        raise ValueError(f"[{section}] must be a table")
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        # Typos like "learning_rat" must fail loudly, not silently use a default.
        raise ValueError(f"Unknown key(s) in [{section}]: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        if "Path" in str(known[key].type) and value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{section}.{key} must be a non-empty path string")
            path = Path(value)
            value = path if path.is_absolute() else (base_dir / path).resolve()
        kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise ValueError(f"[{section}] {error}") from None


def config_from_dict(raw, base_dir="."):
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a table/object")
    base = Path(base_dir).resolve()
    unknown = set(raw) - {"name", "data", "model", "training"}
    if unknown:
        raise ValueError(f"Unknown top-level key(s): {sorted(unknown)}")
    for required in ("name", "data", "model"):
        if required not in raw:
            raise ValueError(f"Missing required key: {required}")
    return ExperimentConfig(
        name=raw["name"],
        data=_build(DataConfig, "data", raw["data"], base),
        model=_build(ModelConfig, "model", raw["model"], base),
        training=_build(TrainingConfig, "training", raw.get("training", {}), base),
    )


def load_config(path):
    path = Path(path)
    if path.suffix == ".toml":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    elif path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Config must be a .toml or .json file")
    return config_from_dict(raw, base_dir=path.parent)
