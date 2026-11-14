"""Prompt construction, report generation, and report parsing.

Research prototype output only. Generated text is not a diagnosis.

Prompt/generation flow adapted from the original experiments, which followed
Unsloth's vision notebooks (https://github.com/unslothai/notebooks).
"""
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import ReportModel

DISCLAIMER = "Research prototype output. Not for diagnosis or clinical use."

# Default prompt for new adapters (used by notebooks/finetune_vlm_colab.ipynb).
# It deliberately does not name a finding: the legacy prompt injected the CNN's
# top label, and the model then described that label instead of the image.
DEFAULT_INSTRUCTION = ("Write a chest X-ray report for this image with FINDINGS and IMPRESSION "
                       "sections. Be concise and do not invent findings.")
SECTIONS = ("TECHNIQUE", "FINDINGS", "IMPRESSION")

_INDENT = " " * 18
_LEGACY_LINES = (
    "You are an expert radiographer. Describe accurately what you see in this image. "
    "the detected disease is {finding}.",
    "",
    "example",
    "",
    "TECHNIQUE: Chest PA and lateral",
    "FINDINGS: The cardiac, mediastinal and hilar",
    "contours are normal. Pulmonary vasculature is",
    "normal. Lungs are clear. No pleural effusion or",
    "pneumothorax is present. Multiple clips are again",
    "seen projecting over the left breast.",
    "Remote left-sided rib fractures are also re-",
    "demonstrated.",
    "IMPRESSION: No acute cardiopulmonary",
    "abnormality.",
    "",
    "Do not use markdown formating. Use lines and keep it concise.",
    "Provide only the answer and it should be aligned with the detected disease.",
    "Only provide the answer and nothing else.",
    "Must include TECHNIQUE, FINDINGS, IMPRESSION.",
)
# Byte-identical to the instruction column of the legacy fine-tuning CSV
# (including its indentation and typos). Kept only to prompt or audit adapters
# trained on that legacy data; new work uses DEFAULT_INSTRUCTION.
LEGACY_INSTRUCTION_TEMPLATE = (
    "\n" + "\n".join(_INDENT + line if line else "" for line in _LEGACY_LINES) + "\n" + " " * 14
)

# Phrases from the prompt's example report. In the legacy synthetic training
# targets they were copied into most reports regardless of the image, so their
# presence in output signals template copying rather than image findings.
EXAMPLE_LEAK_PHRASES = ("multiple clips", "left breast", "rib fractures")

_ROLE_PREFIX = re.compile(r"^\s*(?:user\b\s*)?(?:assistant\b\s*)?", re.IGNORECASE)
_HEADER = re.compile(r"^[ \t]*(TECHNIQUE|FINDINGS|IMPRESSION)[ \t]*:", re.IGNORECASE | re.MULTILINE)


def build_instruction(findings, template=LEGACY_INSTRUCTION_TEMPLATE):
    if isinstance(findings, str):
        findings = [findings]
    findings = [f.strip() for f in findings]
    if not findings or any(not f or "\n" in f for f in findings):
        raise ValueError("Findings must be non-empty single-line labels")
    return template.replace("{finding}", ", ".join(findings))


def build_messages(instruction):
    # Image placeholder first, then text — the order used at legacy inference.
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]


def load_image(path):
    from PIL import Image  # lazy: optional dependency

    with Image.open(Path(path)) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.5
    min_p: float = 0.1
    do_sample: bool = True
    seed: int | None = None

    def __post_init__(self):
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) \
                or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.do_sample and not self.temperature > 0:
            raise ValueError("temperature must be > 0 when sampling")
        if not 0 <= self.min_p < 1:
            raise ValueError("min_p must be in [0, 1)")

    def generate_kwargs(self):
        kwargs = {"max_new_tokens": self.max_new_tokens, "use_cache": True,
                  "do_sample": self.do_sample}
        if self.do_sample:
            kwargs.update(temperature=self.temperature, min_p=self.min_p)
        return kwargs


@dataclass(frozen=True)
class ParsedReport:
    sections: dict
    missing: list
    leaked_example_phrases: list

    @property
    def complete(self):
        return not self.missing


def parse_report(text):
    """Split generated text into TECHNIQUE / FINDINGS / IMPRESSION.

    Strips the stray "user ... assistant" chat-role prefix present in every legacy
    training target. Wrapped lines inside a section are joined with spaces. A
    repeated header keeps the first occurrence.
    """
    text = _ROLE_PREFIX.sub("", text, count=1)
    matches = list(_HEADER.finditer(text))
    sections = {}
    for match, nxt in zip(matches, matches[1:] + [None]):
        name = match.group(1).upper()
        body = text[match.end():nxt.start() if nxt else len(text)]
        if name not in sections:
            sections[name] = " ".join(body.split())
    missing = [s for s in SECTIONS if not sections.get(s)]
    lower = text.lower()
    leaked = [p for p in EXAMPLE_LEAK_PHRASES if p in lower]
    return ParsedReport(sections, missing, leaked)


def generate_text(report_model: ReportModel, image, instruction, settings=GenerationSettings()):
    """Run the VLM on one image and return only the newly generated text."""
    processor = report_model.processor
    if settings.seed is not None:
        import torch  # lazy

        torch.manual_seed(settings.seed)
    prompt = processor.apply_chat_template(build_messages(instruction), add_generation_prompt=True)
    inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(report_model.device)
    prompt_length = inputs["input_ids"].shape[-1]
    output = report_model.model.generate(**inputs, **settings.generate_kwargs())
    # Decode only new tokens; the legacy str.replace(instruction, "") on the full
    # decode was fragile because chat templates alter the echoed prompt.
    return processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()


@dataclass
class ReportResult:
    image: str
    instruction: str
    raw_text: str
    parsed: ParsedReport
    model_id: str
    revision: str | None
    adapter_path: str | None
    settings: GenerationSettings
    disclaimer: str = DISCLAIMER

    def to_dict(self):
        return asdict(self)


def generate_report(image_path, report_model, instruction=DEFAULT_INSTRUCTION,
                    settings=GenerationSettings()):
    """Generate and parse a report for one image."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    image = load_image(image_path)
    raw = generate_text(report_model, image, instruction, settings)
    return ReportResult(str(image_path), instruction, raw, parse_report(raw),
                        report_model.model_id, report_model.revision, report_model.adapter_path,
                        settings)
