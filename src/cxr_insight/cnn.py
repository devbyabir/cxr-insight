"""EfficientNet-B0 multi-label classifier for the 14 NIH ChestX-ray14 findings.

One preprocessing definition is shared by training, evaluation and the demo,
so the deployed model sees inputs exactly as it was trained on them.
Augmentation is applied only for the "train" split, and never mirrors images:
left/right anatomy (heart, aortic knob, situs) is clinically meaningful.

Requires torch and torchvision (imported at module load).
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .nih import NIH_LABELS, parse_labels, to_multi_hot

IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def transform_for(split):
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    return train_transform() if split == "train" else eval_transform()


def load_rgb(source):
    with Image.open(source) as image:
        return image.convert("RGB")


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, rows, image_dir, split):
        self.rows, self.image_dir = list(rows), Path(image_dir)
        self.transform = transform_for(split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.transform(load_rgb(self.image_dir / row["image_path"]))
        target = torch.tensor(to_multi_hot(parse_labels(row["labels"] or "No Finding")),
                              dtype=torch.float32)
        return image, target


def read_split(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def build_model(pretrained=True, n_labels=len(NIH_LABELS)):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_labels)
    return model


def save_classifier(path, model, thresholds, metadata):
    torch.save({"state_dict": model.state_dict(), "labels": list(NIH_LABELS),
                "thresholds": list(thresholds), "architecture": "efficientnet_b0",
                "image_size": IMAGE_SIZE, "metadata": metadata}, path)


def load_classifier(path, map_location="cpu"):
    # weights_only=True: tensors and plain containers only, no arbitrary pickled objects.
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("architecture") != "efficientnet_b0" or \
            tuple(checkpoint.get("labels", ())) != NIH_LABELS:
        raise ValueError("Checkpoint is not an EfficientNet-B0 NIH-14 classifier")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_probabilities(model, image):
    batch = eval_transform()(image.convert("RGB")).unsqueeze(0)
    return torch.sigmoid(model(batch))[0].tolist()


def grad_cam(model, image, label_index):
    """Grad-CAM heatmap (H x W, values 0..1) on the last convolutional block."""
    activations, gradients = {}, {}

    def capture(module, inputs, output):
        activations["a"] = output
        output.register_hook(lambda grad: gradients.__setitem__("g", grad))

    forward = model.features[-1].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        batch = eval_transform()(image.convert("RGB")).unsqueeze(0).requires_grad_(True)
        model(batch)[0, label_index].backward()
        weights = gradients["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["a"]).sum(dim=1))[0]
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.detach().numpy()
    finally:
        forward.remove()
