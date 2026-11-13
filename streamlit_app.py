"""CXR Insight — Streamlit research demo (CPU).

Uploads are processed in memory and never written to disk by this app.
Run locally:  CXR_CLASSIFIER_PATH=runs/<run>/classifier.pt streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cxr_insight.demo import (DISCLAIMER, MAX_UPLOAD_MB, UploadError, draft_report, flagged,  # noqa: E402
                              overlay, resolve_checkpoint, validate_upload)

st.set_page_config(page_title="CXR Insight", page_icon="🩻", layout="wide")


def secret(name):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file configured
        return None


@st.cache_resource(show_spinner="Loading classifier…")
def load_model(path_str):
    from cxr_insight.cnn import load_classifier
    return load_classifier(path_str)


st.title("CXR Insight")
st.caption("Chest X-ray multi-label classification with Grad-CAM and a template draft report.")
st.error(f"**{DISCLAIMER}** Outputs may be wrong. Do not use them for any medical decision.",
         icon="⚠️")

try:
    checkpoint_path = resolve_checkpoint(path=secret("CXR_CLASSIFIER_PATH"),
                                         url=secret("CXR_CLASSIFIER_URL"),
                                         sha256=secret("CXR_CLASSIFIER_SHA256"))
except Exception as error:  # misconfiguration must be visible, not silent
    st.error(f"Model configuration error: {error}")
    st.stop()

if checkpoint_path is None:
    st.warning("No classifier checkpoint is configured, so no predictions can be shown. "
               "Set CXR_CLASSIFIER_PATH, or CXR_CLASSIFIER_URL with CXR_CLASSIFIER_SHA256.")
    st.stop()

try:
    model, checkpoint = load_model(str(checkpoint_path))
except Exception as error:
    st.error(f"The classifier could not be loaded: {error}")
    st.stop()

labels, thresholds = checkpoint["labels"], checkpoint["thresholds"]
meta = checkpoint.get("metadata", {})
data = meta.get("data", {})
with st.sidebar:
    st.subheader("Model")
    st.write(f"EfficientNet-B0, 14 NIH ChestX-ray14 findings (multi-label).")
    if meta.get("best_val_macro_auroc") is not None:
        st.write(f"Validation macro AUROC: **{meta['best_val_macro_auroc']:.3f}**")
    if data.get("subset"):
        st.write(f"Trained on a **subset** of NIH ChestX-ray14 "
                 f"({data.get('images_used')} images), not the full dataset.")
    git = meta.get("git") or {}
    if git.get("commit"):
        note = " plus uncommitted changes at training time" if git.get("dirty") else ""
        st.write(f"Trained from commit `{git['commit'][:10]}`{note}. See the README for the "
                 "commit that reproduces this run.")
    st.caption("Thresholds were fitted on the validation split. See the README for test-set "
               "results and limitations.")
    st.caption("Language-model report generation needs a GPU and is not run in this free "
               "CPU demo.")

upload = st.file_uploader(f"Upload a frontal chest X-ray (PNG or JPEG, ≤ {MAX_UPLOAD_MB} MB)",
                          type=["png", "jpg", "jpeg"])
if upload is None:
    st.info("Upload an image to see predictions. Nothing is stored.")
    st.stop()

try:
    image = validate_upload(upload.getvalue())
except UploadError as error:
    st.error(str(error))
    st.stop()

from cxr_insight.cnn import grad_cam, predict_probabilities  # noqa: E402

with st.spinner("Running the classifier…"):
    probabilities = predict_probabilities(model, image)

hits = flagged(probabilities, labels, thresholds)
left, right = st.columns(2)
with left:
    st.subheader("Image")
    ranked = sorted(range(len(labels)), key=lambda i: probabilities[i], reverse=True)
    choice = st.selectbox("Grad-CAM for finding", [labels[i] for i in ranked],
                          help="Shows which regions most increased this finding's score. "
                               "It is not a localisation of disease.")
    show_cam = st.toggle("Show Grad-CAM overlay", value=True)
    if show_cam:
        with st.spinner("Computing Grad-CAM…"):
            cam = grad_cam(model, image, labels.index(choice))
        st.image(overlay(image, cam), width="stretch",
                 caption=f"Grad-CAM: {choice}")
    else:
        st.image(image, width="stretch")

with right:
    st.subheader("Classifier scores")
    st.dataframe(
        [{"Finding": labels[i].replace("_", " "), "Score": round(probabilities[i], 3),
          "Threshold": round(thresholds[i], 3),
          "Flagged": "yes" if probabilities[i] >= thresholds[i] else ""} for i in ranked],
        hide_index=True, width="stretch")
    st.caption("Scores are independent per finding (sigmoid), so they need not sum to 1.")
    st.subheader("Draft report")
    st.text(draft_report(probabilities, labels, thresholds))

st.divider()
st.caption(f"{DISCLAIMER} Data: NIH ChestX-ray14 (NIH Clinical Center; Wang et al., CVPR 2017).")
