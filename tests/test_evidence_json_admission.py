"""Constructed JSON admission checks; no tensor backend or outcomes are loaded."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import analyze_trajectory_isolated_ablations as analysis
from scripts import verify_delayed_recall_successor_v2 as successor

ROOT = Path(__file__).resolve().parents[1]

DUPLICATES = (
    b'{"execution_authorized":true,"execution_authorized":false}',
    b'{"runs":[{"status":"failed","status":"success"}]}',
    b'{"runs":[{"rollout_mse":8.0,"rollout_mse":0.0}]}',
    b'{"budget":{"epochs":4,"epochs":4}}',
    b'{"runs":[{"status":"failed","\\u0073tatus":"success"}]}',
    b'{"outer":[[{"nested":{"flag":true,"flag":false}}]]}',
)
NONFINITE = (b"NaN", b"Infinity", b"-Infinity", b"1e9999", b"-1e9999")


class EvidenceJsonAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "ARTIFICIAL_INPUT.json"
        self.outputs = [self.root / name for name in ("paired.csv", "table.md", "evidence.md")]
        for destination in self.outputs:
            destination.write_bytes(b"retained artificial output\n")

    def run_reader(self, module, raw: bytes):
        self.source.write_bytes(raw)
        if module is analysis:
            argv = [
                "analyzer", "--manifest", str(self.source), "--csv", str(self.outputs[0]),
                "--table", str(self.outputs[1]), "--evidence", str(self.outputs[2]),
            ]
            boundary = "validate_manifest"
        else:
            argv = ["verifier", "--protocol", str(self.source)]
            boundary = "verify_protocol"
        # A parser rejection must happen before any semantic verifier/backend,
        # comparison function, or output-publication path is reached.
        with patch.object(sys, "argv", argv), patch.object(
            module, boundary,
            side_effect=AssertionError("ambiguous JSON reached semantic admission"),
        ) as semantic:
            with self.assertRaises(ValueError):
                module.main()
            semantic.assert_not_called()
        self.assertTrue(all(path.read_bytes() == b"retained artificial output\n" for path in self.outputs))
        self.assertEqual({p.name for p in self.root.iterdir()}, {self.source.name, *(p.name for p in self.outputs)})

    def test_analysis_rejects_duplicate_fields_before_admission(self) -> None:
        for raw in DUPLICATES:
            with self.subTest(raw=raw):
                self.run_reader(analysis, raw)

    def test_successor_rejects_duplicate_fields_before_admission(self) -> None:
        for raw in DUPLICATES:
            with self.subTest(raw=raw):
                self.run_reader(successor, raw)

    def test_analysis_rejects_nonfinite_values_in_unused_metadata(self) -> None:
        for token in NONFINITE:
            with self.subTest(token=token):
                self.run_reader(analysis, b'{"unused_metadata":' + token + b"}")

    def test_successor_rejects_nonfinite_values_before_backend_loading(self) -> None:
        for token in NONFINITE:
            with self.subTest(token=token):
                self.run_reader(successor, b'{"unused_metadata":' + token + b"}")

    def test_both_readers_reject_nonobject_documents(self) -> None:
        for module in (analysis, successor):
            for raw in (b"[]", b"null", b'"fixture"', b"1", b"true"):
                with self.subTest(module=module.__name__, raw=raw):
                    self.run_reader(module, raw)

    def test_both_readers_reject_invalid_utf8_or_syntax(self) -> None:
        for module in (analysis, successor):
            for raw in (b'{"note":"\xff"}', b'{"incomplete":', b'{}{}'):
                with self.subTest(module=module.__name__, raw=raw):
                    self.run_reader(module, raw)

    def test_analysis_preserves_valid_values_and_exact_source_hash(self) -> None:
        raw = (
            b'{\n "protocol_sha256":"ARTIFICIAL_TEST_ONLY", "counter":9007199254740993,'
            b' "tiny":5e-324, "signed_zero":-0.0, "note":"caf\\u00e9",'
            b' "rows":[{"same":1},{"same":2}], "flag":false, "absent":null\n}\n'
        )
        self.source.write_bytes(raw)
        argv = [
            "analyzer", "--manifest", str(self.source), "--csv", str(self.outputs[0]),
            "--table", str(self.outputs[1]), "--evidence", str(self.outputs[2]),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            analysis, "validate_manifest", return_value=[{"git_commit": "ARTIFICIAL_TEST_ONLY"}],
        ) as semantic, patch.object(
            analysis, "build_comparisons", return_value=[{"fixture": 0.0}],
        ), patch.object(analysis, "render_table", return_value="ARTIFICIAL TABLE\n"), patch.object(
            analysis, "publish_evidence",
        ) as publisher, redirect_stdout(io.StringIO()):
            analysis.main()
        semantic.assert_called_once_with(json.loads(raw))
        parsed = semantic.call_args.args[0]
        self.assertEqual(parsed["counter"], 9007199254740993)
        self.assertEqual(parsed["tiny"], float.fromhex("0x0.0000000000001p-1022"))
        self.assertEqual(parsed["signed_zero"].hex(), "-0x0.0p+0")
        publisher.assert_called_once()
        self.assertEqual(publisher.call_args.kwargs["protected_inputs"], [self.source])
        for _, content in publisher.call_args.args[0][1:]:
            self.assertIn(hashlib.sha256(raw).hexdigest(), content)
        self.assertTrue(all(path.read_bytes() == b"retained artificial output\n" for path in self.outputs))

    def test_successor_preserves_the_complete_unchanged_method(self) -> None:
        # Reading the frozen method is metadata access, not execution/outcome access.
        raw = (ROOT / "research/protocols/FIM_DELAYED_RECALL_SUCCESSOR_V2.json").read_bytes()
        self.source.write_bytes(raw)
        with patch.object(sys, "argv", ["verifier", "--protocol", str(self.source)]), patch.object(
            successor, "verify_protocol", return_value={"execution_authorized": False},
        ) as semantic, redirect_stdout(io.StringIO()):
            successor.main()
        semantic.assert_called_once_with(json.loads(raw))
        successor.verify_frozen_contract(semantic.call_args.args[0])

    def test_standalone_script_invocations_use_strict_admission(self) -> None:
        self.source.write_bytes(b'{"status":"failed","status":"success"}')
        for name, flag in (
            ("analyze_trajectory_isolated_ablations.py", "--manifest"),
            ("verify_delayed_recall_successor_v2.py", "--protocol"),
        ):
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / name), flag, str(self.source)],
                    cwd=self.root, capture_output=True, text=True, timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("duplicate JSON key", result.stderr)
                self.assertNotIn("No module named", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
