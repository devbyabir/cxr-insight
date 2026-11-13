import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cxr_insight.config import config_from_dict, load_cnn_config, load_config


def minimal():
    return {"name": "exp",
            "data": {"manifest": "m.csv", "image_root": "img", "output_dir": "out"},
            "model": {"base_model_id": "org/model"}}


class ConfigTests(unittest.TestCase):
    def test_vlm_config_loads(self):
        cfg = load_config(ROOT / "configs" / "vlm_qwen2vl_2b.toml")
        self.assertEqual(cfg.model.base_model_id, "unsloth/Qwen2-VL-2B-Instruct")
        self.assertEqual(cfg.training.lora_r, 16)
        self.assertIsNone(cfg.model.revision)

    def test_cnn_run_config_matches_reported_run(self):
        cfg = load_cnn_config(ROOT / "configs" / "cnn_nih_images001.toml")
        self.assertEqual(cfg.splits, (ROOT / "data" / "splits" / "nih_images001_seed42").resolve())
        meta = json.loads((ROOT / "docs" / "results" / "cnn_nih_images001_seed42" /
                           "run_metadata.json").read_text(encoding="utf-8"))["args"]
        self.assertEqual((str(cfg.epochs), str(cfg.batch_size), str(cfg.lr), str(cfg.weight_decay),
                          str(cfg.seed), str(cfg.workers), str(not cfg.pretrained)),
                         (meta["epochs"], meta["batch_size"], meta["lr"], meta["weight_decay"],
                          meta["seed"], meta["workers"], meta["no_pretrained"]))

    def test_cnn_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            good = 'name = "x"\n[cnn]\nsplits = "s"\nimage_dir = "i"\nout = "o"\n'
            path.write_text(good, encoding="utf-8")
            self.assertEqual(load_cnn_config(path).out, (Path(tmp) / "o").resolve())
            for extra in ("epochs = 0", "lr = 0", "pretrained = 1", "batch_size = 2.5", "typo = 1"):
                with self.subTest(extra=extra):
                    path.write_text(good + extra + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_cnn_config(path)
            path.write_text('name = "x"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cnn_config(path)

    def test_defaults_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_from_dict(minimal(), base_dir=tmp)
            self.assertEqual(cfg.data.image_root, (Path(tmp) / "img").resolve())
            self.assertEqual(cfg.data.seed, 42)
            self.assertEqual(cfg.training.epochs, 1)
            self.assertIsNone(cfg.model.adapter_path)

    def test_json_and_toml_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = Path(tmp) / "c.json"
            j.write_text(json.dumps(minimal()), encoding="utf-8")
            t = Path(tmp) / "c.toml"
            t.write_text('name = "exp"\n[data]\nmanifest = "m.csv"\nimage_root = "img"\n'
                         'output_dir = "out"\n[model]\nbase_model_id = "org/model"\n', encoding="utf-8")
            self.assertEqual(load_config(j), load_config(t))

    def test_to_dict_is_json_serialisable(self):
        d = config_from_dict(minimal()).to_dict()
        self.assertEqual(d["model"]["base_model_id"], "org/model")
        self.assertIsInstance(d["data"]["manifest"], str)

    def test_missing_required(self):
        for key in ("name", "data", "model"):
            raw = minimal(); del raw[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                config_from_dict(raw)
        raw = minimal(); del raw["data"]["manifest"]
        with self.assertRaises(ValueError):
            config_from_dict(raw)

    def test_unknown_keys_rejected(self):
        raw = minimal(); raw["training"] = {"learning_rat": 1e-4}
        with self.assertRaises(ValueError):
            config_from_dict(raw)
        raw = minimal(); raw["extra"] = 1
        with self.assertRaises(ValueError):
            config_from_dict(raw)

    def test_invalid_values_rejected(self):
        cases = [("data", "validation", 0.9), ("data", "test", 0), ("data", "seed", -1),
                 ("data", "validation", "0.1"), ("data", "manifest", ""),
                 ("model", "base_model_id", " "), ("model", "revision", ""),
                 ("training", "learning_rate", 0), ("training", "learning_rate", True),
                 ("training", "epochs", 0), ("training", "batch_size", 2.5),
                 ("training", "lora_r", True)]
        for section, key, value in cases:
            raw = minimal(); raw.setdefault(section, {})[key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                config_from_dict(raw)

    def test_split_fractions_match_data_rules(self):
        from cxr_insight.data import split_patients
        cfg = config_from_dict(minimal())
        rows = [dict(patient_id=f"p{i}", sha256=str(i)) for i in range(20)]
        split_patients(rows, cfg.data.seed, cfg.data.validation, cfg.data.test)

    def test_unsupported_extension(self):
        with self.assertRaises(ValueError):
            load_config("config.yaml")


if __name__ == "__main__":
    unittest.main()
