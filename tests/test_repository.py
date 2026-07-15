from __future__ import annotations

import json
import re
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
        self.assertIn("m008_c055", config["competition"]["artifact_name"])
        self.assertAlmostEqual(sum(item["weight"] for item in config["sim"]["models"]), 1.0)
        self.assertAlmostEqual(sum(item["weight"] for item in config["au"]["models"]), 1.0)
        gate = config["sim"]["klue_tiebreak"]
        self.assertEqual(gate["blend_top2_margin_lt"], 0.08)
        self.assertEqual(gate["klue_confidence_gte"], 0.55)

    def test_highscore_manifest_matches_public_contract(self) -> None:
        stack = json.loads((ROOT / "configs" / "final_stack.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (ROOT / "reference" / "inference" / "build_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["result_status"], "confirmed")
        self.assertEqual(manifest["public_score"], stack["competition"]["public_score"])
        self.assertEqual(manifest["artifact_name"], stack["competition"]["artifact_name"])
        conditions = manifest["postprocess"]["klue_tiebreak"]["conditions"]
        self.assertIn("margin < 0.08", conditions)
        self.assertIn("confidence >= 0.55", conditions)

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

    def test_relative_markdown_links_resolve(self) -> None:
        link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        broken = []
        for document in ROOT.rglob("*.md"):
            for target in link_pattern.findall(document.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                relative_target = target.split("#", 1)[0]
                if relative_target and not (document.parent / relative_target).resolve().exists():
                    broken.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual(broken, [])


if __name__ == "__main__":
    unittest.main()
