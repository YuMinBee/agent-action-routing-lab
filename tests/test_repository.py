from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTest(unittest.TestCase):
    def test_final_stack_contract(self) -> None:
        config = json.loads((ROOT / "configs" / "final_stack.json").read_text(encoding="utf-8"))
        self.assertEqual(len(config["labels"]), 14)
        self.assertEqual(len(set(config["labels"])), 14)
        self.assertAlmostEqual(config["competition"]["public_score"], 0.7899327253)
        self.assertEqual(config["competition"]["rank"], 46)
        self.assertAlmostEqual(sum(item["weight"] for item in config["sim"]["models"]), 1.0)
        self.assertAlmostEqual(sum(item["weight"] for item in config["au"]["models"]), 1.0)
        gate = config["sim"]["klue_tiebreak"]
        self.assertEqual(gate["blend_top2_margin_lt"], 0.08)
        self.assertEqual(gate["klue_confidence_gte"], 0.55)

    def test_class_bias_matches_label_order(self) -> None:
        stack = json.loads((ROOT / "configs" / "final_stack.json").read_text(encoding="utf-8"))
        bias = json.loads((ROOT / "configs" / "class_bias_final.json").read_text(encoding="utf-8"))
        self.assertEqual(stack["labels"], bias["labels"])
        self.assertEqual(len(bias["bias_sim"]), 14)
        self.assertEqual(len(bias["bias_au"]), 14)

    def test_no_model_artifacts(self) -> None:
        forbidden = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".npy", ".npz"}
        found = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix in forbidden]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
