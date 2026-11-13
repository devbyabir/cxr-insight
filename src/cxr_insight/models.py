"""Model wrappers: CNN finding classifier and vision-language report model.

Heavy frameworks (TensorFlow, torch, Unsloth) are imported lazily inside the
loader functions, so this module imports — and its pure logic is testable —
without them. Loading the report model requires a CUDA GPU (e.g. Colab).

The vision-language loading pattern is adapted from Unsloth's public vision
fine-tuning notebooks (https://github.com/unslothai/notebooks), which the
original experiments were based on. Retain that attribution, and check the
upstream notebook license, before redistributing derived code.
"""
from dataclasses import dataclass
from pathlib import Path

# Label order of the legacy EfficientNet/ResNet checkpoints (NIH ChestX-ray14
# labels plus "No Finding"). The checkpoint's output index i maps to LABELS[i];
# a different checkpoint needs its own label list.
LABELS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
)
CLASSIFIER_INPUT_SIZE = (224, 224)


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


def rank_predictions(scores, labels=LABELS):
    """Pair raw model outputs with labels, highest score first."""
    scores = [float(s) for s in scores]
    if len(scores) != len(labels):
        raise ValueError(f"Expected {len(labels)} scores, got {len(scores)}")
    return sorted((Prediction(l, s) for l, s in zip(labels, scores)),
                  key=lambda p: p.score, reverse=True)


def select_findings(predictions, threshold=None):
    """Return the finding label(s) used to condition the report prompt.

    threshold=None reproduces the legacy single-label argmax behaviour. With a
    threshold, every label at or above it is kept (multi-label; ChestX-ray14
    findings can coexist), falling back to the top label if none qualify.
    "No Finding" is never combined with positive findings.
    """
    if not predictions:
        raise ValueError("No predictions")
    ranked = sorted(predictions, key=lambda p: p.score, reverse=True)
    if threshold is None:
        return [ranked[0].label]
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    kept = [p.label for p in ranked if p.score >= threshold]
    positives = [label for label in kept if label != "No Finding"]
    return positives or kept[:1] or [ranked[0].label]


def prepare_classifier_input(image, size=CLASSIFIER_INPUT_SIZE):
    """RGB, resize, uint8 HWC array with a batch axis — as the legacy notebooks did.

    No rescaling is applied: Keras EfficientNet models include their own
    preprocessing layer. A checkpoint expecting normalised input differs.
    """
    import numpy as np

    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.expand_dims(np.asarray(image.resize(size)), 0)


class FindingClassifier:
    """Wraps any model exposing Keras-style ``predict(batch) -> [[scores]]``."""

    def __init__(self, model, labels=LABELS, input_size=CLASSIFIER_INPUT_SIZE):
        self.model, self.labels, self.input_size = model, tuple(labels), input_size

    @classmethod
    def from_keras(cls, path, **kwargs):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
        from tensorflow.keras.models import load_model  # lazy: heavy dependency

        return cls(load_model(path), **kwargs)

    def predict(self, image):
        batch = prepare_classifier_input(image, self.input_size)
        output = self.model.predict(batch, verbose=0)
        return rank_predictions(output[0], self.labels)


@dataclass
class ReportModel:
    """A loaded vision-language model and its processor/tokenizer."""
    model: object
    processor: object
    model_id: str
    revision: str | None = None
    adapter_path: str | None = None
    device: str = "cuda"


def load_report_model(model_id, revision=None, adapter_path=None, load_in_4bit=True):
    """Load a base VLM, or a trained LoRA adapter on top of it, for inference.

    Unlike the legacy inference notebooks, this never calls
    ``FastVisionModel.get_peft_model`` at inference time: that attaches a
    *freshly initialised, untrained* adapter, so those notebooks were in effect
    evaluating the base model. Pass ``adapter_path`` to use trained weights.
    """
    from unsloth import FastVisionModel  # lazy: requires CUDA

    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"No LoRA adapter (adapter_config.json) in {adapter_path}")
        # Unsloth resolves the base model from adapter_config.json.
        source, kwargs = str(adapter_path), {}
    else:
        source, kwargs = model_id, ({"revision": revision} if revision else {})
    model, processor = FastVisionModel.from_pretrained(source, load_in_4bit=load_in_4bit, **kwargs)
    FastVisionModel.for_inference(model)
    return ReportModel(model, processor, model_id, revision,
                       str(adapter_path) if adapter_path else None)


def verify_trained_adapter(model):
    """Confirm LoRA weights are present *and trained*; return a summary.

    A freshly initialised LoRA adapter has every ``lora_B`` matrix equal to zero
    (so it changes nothing). Training makes them non-zero. Raises if the model
    has no LoRA parameters or all ``lora_B`` weights are still zero.
    """
    lora_a = lora_b = nonzero_b = 0
    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_a += 1
        elif "lora_B" in name:
            lora_b += 1
            nonzero_b += float(param.detach().abs().sum()) > 0
    if lora_b == 0:
        raise RuntimeError("No LoRA parameters found: the adapter was not loaded")
    if nonzero_b == 0:
        raise RuntimeError("All lora_B weights are zero: this adapter is untrained")
    return {"lora_A_tensors": lora_a, "lora_B_tensors": lora_b, "nonzero_lora_B_tensors": nonzero_b}


def load_report_model_from_config(model_config, load_in_4bit=True):
    return load_report_model(model_config.base_model_id, model_config.revision,
                             model_config.adapter_path, load_in_4bit)
