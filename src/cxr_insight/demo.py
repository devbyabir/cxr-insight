"""Framework-free helpers for the Streamlit demo (upload checks, draft text, overlay).

The demo's draft report is a deterministic template filled from classifier
scores. No language model runs in the free CPU deployment, and the text says so.
"""
import hashlib
import io
import os
import urllib.request
from pathlib import Path

DISCLAIMER = "Research demo — not for clinical diagnosis."
MAX_UPLOAD_MB = 10
MIN_SIDE = 128


class UploadError(ValueError):
    pass


def validate_upload(data, max_mb=MAX_UPLOAD_MB):
    """Return an RGB PIL image from uploaded bytes, kept in memory only."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(f"The file is larger than {max_mb} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise UploadError("Please upload a PNG or JPEG image.")
            if min(image.size) < MIN_SIDE:
                raise UploadError(f"The image is too small (minimum {MIN_SIDE}px per side).")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise UploadError("The file could not be read as an image.") from error


def flagged(probabilities, labels, thresholds):
    """Labels at/above their validation-fitted threshold, highest score first."""
    hits = [(label, p) for label, p, t in zip(labels, probabilities, thresholds) if p >= t]
    return sorted(hits, key=lambda item: item[1], reverse=True)


def draft_report(probabilities, labels, thresholds):
    hits = flagged(probabilities, labels, thresholds)
    if hits:
        listed = "; ".join(f"{label.replace('_', ' ')} (score {p:.2f})" for label, p in hits)
        findings = f"Classifier scores exceed the validation threshold for: {listed}."
        impression = ("Possible " + ", ".join(label.replace("_", " ").lower() for label, _ in hits)
                      + ". Unconfirmed automated output; requires review by a radiologist.")
    else:
        findings = "No finding exceeded its classifier threshold."
        impression = ("No automated flag. This does not exclude disease; "
                      "requires review by a radiologist.")
    return "\n".join([
        "TECHNIQUE: Single uploaded chest radiograph (projection not verified).",
        f"FINDINGS: {findings}",
        f"IMPRESSION: {impression}",
        "",
        "Template draft generated from classifier scores; no language model was used. "
        + DISCLAIMER,
    ])


def overlay(image, cam, alpha=0.4):
    """Blend a 0..1 heatmap (any size) over an RGB PIL image."""
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image

    cam = np.clip(np.asarray(cam, dtype=float), 0, 1)
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    return Image.fromarray(np.uint8(((1 - alpha) * base + alpha * colored) * 255))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(path=None, url=None, sha256=None, cache_dir=None):
    """Find the classifier: a local path, or a URL downloaded once and hash-checked.

    Returns None when nothing is configured, so the UI can say so honestly.
    """
    path = path or os.environ.get("CXR_CLASSIFIER_PATH")
    url = url or os.environ.get("CXR_CLASSIFIER_URL")
    sha256 = sha256 or os.environ.get("CXR_CLASSIFIER_SHA256")
    if path:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Configured checkpoint not found: {path}")
    elif url:
        if not url.startswith("https://"):
            raise ValueError("Checkpoint URL must use https")
        if not sha256:
            raise ValueError("CXR_CLASSIFIER_SHA256 is required with a checkpoint URL")
        cache = Path(cache_dir or Path.home() / ".cache" / "cxr-insight")
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"classifier-{sha256[:16]}.pt"
        if not path.is_file():
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(url, partial)
            partial.replace(path)
    else:
        return None
    if sha256 and sha256_file(path) != sha256.lower():
        raise ValueError("Checkpoint hash mismatch; refusing to load it")
    return path
