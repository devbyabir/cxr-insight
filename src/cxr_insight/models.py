"""Vision-language report model: loading trained adapters and checking them.

Unsloth/torch are imported lazily inside the loader, so this module imports
without them. Loading a report model requires a CUDA GPU (e.g. Colab).
The CNN classifier lives in ``cxr_insight.cnn``.

The vision-language loading pattern is adapted from Unsloth's public vision
fine-tuning notebooks (https://github.com/unslothai/notebooks), which the
original experiments were based on. Retain that attribution, and check the
upstream notebook license, before redistributing derived code.
"""
from dataclasses import dataclass
from pathlib import Path


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
