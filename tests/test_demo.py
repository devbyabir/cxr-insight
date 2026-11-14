import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_insight.demo import (DISCLAIMER, MAX_PIXELS, UploadError, download, draft_report, flagged,
                              resolve_checkpoint, sha256_file, validate_upload)

HAS_IMAGING = all(importlib.util.find_spec(m) for m in ("numpy", "PIL", "matplotlib"))
LABELS = ("Effusion", "Mass", "Pleural_Thickening")


def encoded(fmt="PNG", size=(256, 256), mode="L"):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new(mode, size, 120).save(buffer, fmt)
    return buffer.getvalue()


class DraftTests(unittest.TestCase):
    def test_flagged_uses_per_label_thresholds(self):
        self.assertEqual(flagged([0.6, 0.4, 0.35], LABELS, [0.5, 0.5, 0.3]),
                         [("Effusion", 0.6), ("Pleural_Thickening", 0.35)])

    def test_draft_with_findings(self):
        text = draft_report([0.6, 0.1, 0.35], LABELS, [0.5, 0.5, 0.3])
        self.assertIn("Effusion (score 0.60); Pleural Thickening (score 0.35)", text)
        self.assertIn("radiologist", text)
        self.assertIn("no language model was used", text)
        self.assertIn(DISCLAIMER, text)
        for section in ("TECHNIQUE:", "FINDINGS:", "IMPRESSION:"):
            self.assertIn(section, text)

    def test_draft_without_findings_does_not_claim_normal(self):
        text = draft_report([0.1, 0.1, 0.1], LABELS, [0.5, 0.5, 0.5])
        self.assertIn("does not exclude disease", text)
        self.assertNotIn("normal", text.lower())


@unittest.skipUnless(HAS_IMAGING, "numpy, Pillow and matplotlib required")
class UploadTests(unittest.TestCase):
    def test_accepts_png_and_jpeg_as_rgb(self):
        for fmt in ("PNG", "JPEG"):
            with self.subTest(fmt=fmt):
                self.assertEqual(validate_upload(encoded(fmt)).mode, "RGB")

    def test_rejections(self):
        cases = [b"", b"not an image", encoded(size=(64, 300)), encoded("BMP")]
        for data in cases:
            with self.subTest(data=data[:12]), self.assertRaises(UploadError):
                validate_upload(data)
        with self.assertRaises(UploadError):
            validate_upload(encoded(), max_mb=0.00001)

    def test_pixel_bombs_rejected_before_decoding(self):
        # Highly compressible images whose decoded size would be huge.
        for side in (7000, 20000):  # above our cap; the second is above Pillow's hard limit
            with self.subTest(side=side), self.assertRaisesRegex(UploadError, "too many pixels"):
                validate_upload(encoded(size=(side, side)), max_mb=10)
        self.assertLess(6000 * 6000, MAX_PIXELS + 1)

    def test_truncated_image_rejected(self):
        with self.assertRaises(UploadError):
            validate_upload(encoded()[:200])

    def test_overlay_matches_image_size(self):
        from PIL import Image
        from cxr_insight.demo import overlay
        out = overlay(Image.new("RGB", (300, 200)), [[0, 1], [0.5, 0.2]])
        self.assertEqual((out.size, out.mode), ((300, 200), "RGB"))


class CheckpointResolutionTests(unittest.TestCase):
    def test_nothing_configured_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_checkpoint())

    def test_local_path_and_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            path.write_bytes(b"weights")
            self.assertEqual(resolve_checkpoint(path=str(path)), path)
            self.assertEqual(resolve_checkpoint(path=str(path), sha256=sha256_file(path)), path)
            with self.assertRaises(ValueError):
                resolve_checkpoint(path=str(path), sha256="0" * 64)
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(path=str(Path(tmp) / "missing.pt"))

    def test_url_requires_https_and_hash(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_checkpoint(url="http://example.com/c.pt", sha256="a" * 64)
            with self.assertRaises(ValueError):
                resolve_checkpoint(url="https://example.com/c.pt")

    def test_url_download_cached_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = b"weights"
            digest = hashlib.sha256(payload).hexdigest()
            calls = []

            def fake_download(url, dest, **kwargs):
                calls.append(url)
                Path(dest).write_bytes(payload)

            with mock.patch("cxr_insight.demo.download", fake_download):
                first = resolve_checkpoint(url="https://example.com/c.pt", sha256=digest,
                                           cache_dir=Path(tmp) / "cache")
                second = resolve_checkpoint(url="https://example.com/c.pt", sha256=digest,
                                            cache_dir=Path(tmp) / "cache")
                self.assertEqual((first, len(calls)), (second, 1))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    resolve_checkpoint(url="https://example.com/other.pt", sha256="b" * 64,
                                       cache_dir=Path(tmp) / "cache")
                self.assertEqual(list((Path(tmp) / "cache").glob("*.part")), [])


class FakeResponse(io.BytesIO):
    def __init__(self, data, declared=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(declared)} if declared is not None else {}


class DownloadTests(unittest.TestCase):
    def test_streams_within_limit(self):
        with tempfile.TemporaryDirectory() as tmp,                 mock.patch("urllib.request.urlopen", lambda url, timeout: FakeResponse(b"x" * 10)):
            download("https://example.com/c.pt", Path(tmp) / "c.pt", max_mb=1)
            self.assertEqual((Path(tmp) / "c.pt").read_bytes(), b"x" * 10)

    def test_rejects_oversized_declared_and_streamed(self):
        big = b"x" * (1024 * 1024 + 1)
        for declared in (len(big), None):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as tmp,                     mock.patch("urllib.request.urlopen",
                               lambda url, timeout, d=declared: FakeResponse(big, d)):
                with self.assertRaisesRegex(ValueError, "larger than 1 MB"):
                    download("https://example.com/c.pt", Path(tmp) / "c.pt", max_mb=1)
                self.assertFalse((Path(tmp) / "c.pt").exists())


if __name__ == "__main__":
    unittest.main()
