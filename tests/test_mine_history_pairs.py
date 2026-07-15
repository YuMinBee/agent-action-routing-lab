from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_history_pairs.py"
SPEC = importlib.util.spec_from_file_location("mine_history_pairs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class HistoryMiningTest(unittest.TestCase):
    def test_mines_only_missing_past_pair_and_strips_dynamic_state(self) -> None:
        rows = [
            {
                "id": "sess_sim_demo-step_03",
                "current_prompt": "run tests",
                "history": [
                    {"role": "user", "content": "show files"},
                    {"name": "list_directory", "args": {"path": "."}, "result_summary": "3 files"},
                    {"role": "user", "content": "read app.py"},
                    {"name": "read_file", "args": {"path": "app.py"}, "result_summary": "20 lines"},
                ],
                "session_meta": {
                    "language_pref": "en",
                    "user_tier": "pro",
                    "budget_tokens_remaining": 9000,
                    "workspace": {"open_files": ["app.py"]},
                },
            }
        ]
        known = {"sess_sim_demo-step_01": "list_directory"}
        mined, labels, report = MODULE.mine_rows(rows, known)

        self.assertEqual(report["overlap_checked"], 1)
        self.assertEqual(report["overlap_mismatches"], 0)
        self.assertEqual(labels, [{"id": "sess_sim_demo-step_02", "action": "read_file"}])
        self.assertEqual(mined[0]["current_prompt"], "read app.py")
        self.assertEqual(len(mined[0]["history"]), 2)
        self.assertEqual(
            mined[0]["session_meta"], {"language_pref": "en", "user_tier": "pro"}
        )
        self.assertNotIn("workspace", mined[0]["session_meta"])

    def test_conflicting_windows_are_rejected(self) -> None:
        base = {
            "current_prompt": "later",
            "session_meta": {},
        }
        rows = [
            {
                **base,
                "id": "sess_au_demo-step_02",
                "history": [
                    {"role": "user", "content": "find it"},
                    {"name": "grep_search"},
                ],
            },
            {
                **base,
                "id": "sess_au_demo-step_03",
                "history": [
                    {"role": "user", "content": "find it"},
                    {"name": "glob_pattern"},
                ],
            },
        ]
        mined, labels, report = MODULE.mine_rows(rows)
        self.assertEqual(mined, [])
        self.assertEqual(labels, [])
        self.assertEqual(report["conflict_keys"], 1)


if __name__ == "__main__":
    unittest.main()
