"""Unit and integration tests for the Qwen3.5-2B Exam Generator Distillation Pipeline."""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import local_runner
import prepare_dataset
import schema
import validate_dataset


def _make_valid_record(
    *,
    level: str = "SMA",
    lang: str = "id",
    student_body: str | None = None,
    answer_body: str | None = None,
) -> dict:
    user_prompt = schema.build_user_prompt(
        topic="Termodinamika",
        education_level=level,
        bloom_profile="C3-C4",
        question_formats="MCQ",
        num_questions=1,
        language=lang,
    )
    if student_body is None:
        if level in ("SD", "SMP"):
            student_body = (
                "Ujian Fisika\nPetunjuk: Pilih satu jawaban yang paling tepat.\n\n"
                "1. Sebuah gas ideal mengalami proses isotermal pada suhu tetap. "
                "Besaran yang nilainya nol pada proses ini adalah...\n"
                "A. Usaha\nB. Kalor\nC. Perubahan energi dalam\nD. Tekanan"
            )
        else:
            student_body = (
                "Ujian Fisika\nPetunjuk: Pilih satu jawaban yang paling tepat.\n\n"
                "1. Sebuah gas ideal mengalami proses isotermal pada suhu tetap. "
                "Besaran yang nilainya nol pada proses ini adalah...\n"
                "A. Usaha\nB. Kalor\nC. Perubahan energi dalam\nD. Tekanan\nE. Volume"
            )
    if answer_body is None:
        answer_body = (
            "1. Kunci: C\n"
            "Mengapa benar: Pada proses isotermal suhu tetap sehingga perubahan energi dalam nol.\n"
            "- A: Miskonsepsi mengira usaha nol pada suhu tetap.\n"
            "- B: Miskonsepsi mengira kalor tidak berpindah.\n"
            "- D: Miskonsepsi mengira tekanan konstan.\n"
            "- E: Miskonsepsi mengira volume konstan."
        )
    assistant_content = (
        f"{schema.STUDENT_SHEET_START}\n{student_body}\n{schema.STUDENT_SHEET_END}\n"
        f"{schema.ANSWER_KEY_START}\n{answer_body}\n{schema.ANSWER_KEY_END}"
    )
    return {
        "messages": [
            {"role": "system", "content": schema.CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


class TestSchema(unittest.TestCase):
    def test_canonical_system_prompt_tags(self) -> None:
        self.assertIn(schema.STUDENT_SHEET_START, schema.CANONICAL_SYSTEM_PROMPT)
        self.assertIn(schema.STUDENT_SHEET_END, schema.CANONICAL_SYSTEM_PROMPT)
        self.assertIn(schema.ANSWER_KEY_START, schema.CANONICAL_SYSTEM_PROMPT)
        self.assertIn(schema.ANSWER_KEY_END, schema.CANONICAL_SYSTEM_PROMPT)

    def test_expected_mcq_option_count(self) -> None:
        for lvl in ("SD", "SMP", "sd/mi", "elementary", "Junior High"):
            self.assertEqual(schema.expected_mcq_option_count(lvl), 4)
        for lvl in ("SMA", "SMK", "Kuliah", "University", "S1", "College"):
            self.assertEqual(schema.expected_mcq_option_count(lvl), 5)
        with self.assertRaises(ValueError):
            schema.expected_mcq_option_count("UnknownLevelXYZ")
        with self.assertRaises(ValueError):
            schema.expected_mcq_option_count("")

    def test_build_user_prompt_id_and_en(self) -> None:
        p_id = schema.build_user_prompt(
            topic="Aljabar",
            education_level="SMP",
            bloom_profile="C3-C4",
            question_formats="MCQ",
            num_questions=5,
            language="id",
            reference_material="Persamaan linear satu variabel.",
            show_bloom_labels=True,
        )
        self.assertIn("Topik: Aljabar", p_id)
        self.assertIn("Materi Referensi:", p_id)
        self.assertIn("Label Bloom", p_id)

        p_en = schema.build_user_prompt(
            topic="Graphs",
            education_level="University",
            bloom_profile="C4-C6",
            question_formats="Case Study",
            num_questions=2,
            language="en",
            reference_material="BFS and DFS traversal.",
            show_bloom_labels=False,
        )
        self.assertIn("Topic: Graphs", p_en)
        self.assertIn("Reference Material:", p_en)
        self.assertNotIn("Include Bloom", p_en)

    def test_strip_thinking(self) -> None:
        raw = "<think>\nreasoning here\n</think>\n\n<<<STUDENT_SHEET>>>\nQ1\n<<</STUDENT_SHEET>>>"
        cleaned = schema.strip_thinking(raw)
        self.assertNotIn("<think>", cleaned)
        self.assertTrue(cleaned.startswith("<<<STUDENT_SHEET>>>"))

    def test_parse_exam_output_canonical_and_fallback(self) -> None:
        canonical = (
            "<think>ok</think>\n"
            "<<<STUDENT_SHEET>>>\nSoal 1\n<<</STUDENT_SHEET>>>\n"
            "<<<ANSWER_KEY>>>\nKunci 1\n<<</ANSWER_KEY>>>"
        )
        parsed = schema.parse_exam_output(canonical)
        self.assertEqual(parsed["student_sheet"], "Soal 1")
        self.assertEqual(parsed["answer_key"], "Kunci 1")

        fallback = "## Student Sheet\nQuestion 1\n\n## Answer Key\n1. A"
        parsed_fb = schema.parse_exam_output(fallback)
        self.assertEqual(parsed_fb["student_sheet"], "Question 1")
        self.assertEqual(parsed_fb["answer_key"], "1. A")


class TestValidateDataset(unittest.TestCase):
    def test_valid_record_passes(self) -> None:
        rec = _make_valid_record(level="SMA")
        errors, warnings, stats = validate_dataset.validate_record(rec)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertIsInstance(stats, dict)

    def test_leak_detection(self) -> None:
        rec = _make_valid_record(
            level="SMA",
            student_body=(
                "1. Pertanyaan?\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5\n"
                "Kunci: A"
            ),
        )
        errors, _, _ = validate_dataset.validate_record(rec)
        self.assertTrue(any("leak" in e.lower() or "kunci" in e.lower() for e in errors))

    def test_bloom_tag_vs_chemical_formula(self) -> None:
        rec_chem = _make_valid_record(
            level="SMA",
            student_body=(
                "1. Rumus kimia etanol adalah (C2H5OH) dan glukosa (C6H12O6). "
                "Manakah yang benar?\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5"
            ),
        )
        errors_chem, _, _ = validate_dataset.validate_record(rec_chem)
        self.assertFalse(any("Bloom" in e for e in errors_chem))

        rec_bloom = _make_valid_record(
            level="SMA",
            student_body=(
                "1. (C3) Hitunglah nilai x berikut!\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5"
            ),
        )
        errors_bloom, _, _ = validate_dataset.validate_record(rec_bloom)
        self.assertTrue(any("Bloom" in e for e in errors_bloom))

    def test_forbidden_option_phrase(self) -> None:
        rec = _make_valid_record(
            level="SMA",
            student_body=(
                "1. Pilih yang tepat:\nA. 1\nB. 2\nC. 3\nD. 4\nE. Semua jawaban benar"
            ),
        )
        errors, _, _ = validate_dataset.validate_record(rec)
        self.assertTrue(any("Forbidden option" in e for e in errors))

    def test_sd_smp_option_e_fails_and_sma_missing_e_fails(self) -> None:
        rec_smp_e = _make_valid_record(
            level="SMP",
            student_body="1. Soal SMP?\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5",
        )
        errors_smp, _, _ = validate_dataset.validate_record(rec_smp_e)
        self.assertTrue(any("option E" in e or "4 options" in e for e in errors_smp))

        rec_sma_no_e = _make_valid_record(
            level="SMA",
            student_body="1. Soal SMA?\nA. 1\nB. 2\nC. 3\nD. 4",
        )
        errors_sma, _, _ = validate_dataset.validate_record(rec_sma_no_e)
        self.assertTrue(any("missing option E" in e or "5 options" in e for e in errors_sma))

    def test_max_char_check(self) -> None:
        rec = _make_valid_record(
            level="SMA",
            student_body="1. " + ("x" * (validate_dataset.MAX_RECORD_CHARS + 100)) + "\nA. 1\nB. 2\nC. 3\nD. 4\nE. 5",
        )
        errors, _, _ = validate_dataset.validate_record(rec)
        self.assertTrue(any("11500" in e or "char" in e.lower() or "length" in e.lower() for e in errors))

    def test_validate_actual_datasets(self) -> None:
        expected = [
            (ROOT / "dataset_exam_generator.jsonl", 96),
            (ROOT / "dataset_train.jsonl", 86),
            (ROOT / "dataset_val.jsonl", 10),
        ]
        for path, count in expected:
            self.assertTrue(path.exists(), f"Missing dataset file: {path}")
            errors, _, stats = validate_dataset.validate_jsonl(path)
            self.assertEqual(errors, [], f"Errors in {path.name}: {errors}")
            self.assertEqual(stats["total_records"], count)
            self.assertEqual(stats["valid_records"], count)


class TestPrepareDataset(unittest.TestCase):
    def test_shards_and_no_train_val_overlap(self) -> None:
        shards = sorted((ROOT / "data_batches").glob("*.jsonl"))
        self.assertEqual(len(shards), 8)

        def load_prompts(path: Path) -> list[str]:
            prompts = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        prompts.append(rec["messages"][1]["content"])
            return prompts

        train_prompts = load_prompts(ROOT / "dataset_train.jsonl")
        val_prompts = load_prompts(ROOT / "dataset_val.jsonl")
        self.assertEqual(len(train_prompts), 86)
        self.assertEqual(len(val_prompts), 10)
        self.assertEqual(set(train_prompts) & set(val_prompts), set())


class _FakeHTTPResponse:
    def __init__(self, body_bytes: bytes) -> None:
        self._stream = io.BytesIO(body_bytes)

    def read(self) -> bytes:
        return self._stream.read()

    def __iter__(self):
        return iter(self._stream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestLocalRunner(unittest.TestCase):
    def test_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "local_runner.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--backend", proc.stdout)
        self.assertIn("--eval-val", proc.stdout)

    def test_write_modelfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gguf = tmp / "Qwen3.5-2B.Q4_K_M.gguf"
            gguf.write_bytes(b"gguf")
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.object(sys, "argv", ["local_runner.py", "--write-modelfile", str(gguf)]):
                    rc = local_runner.main()
            finally:
                os.chdir(old_cwd)
            self.assertEqual(rc, 0)
            modelfile = tmp / "Modelfile"
            self.assertTrue(modelfile.exists())
            content = modelfile.read_text(encoding="utf-8")
            self.assertNotIn("{{GGUF_PATH}}", content)
            self.assertIn(f"FROM {gguf}", content)

    def test_read_reference_file_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "ref.txt"
            p.write_bytes(b"\xef\xbb\xbfMateri Referensi BOM")
            text = local_runner._read_reference_file(str(p))
            self.assertEqual(text, "Materi Referensi BOM")

    def test_ollama_backend_mocked(self) -> None:
        stream_lines = (
            json.dumps({"message": {"content": "<think>x</think><<<STUDENT_SHEET>>>\nQ1\n"}})
            + "\n"
            + json.dumps({"message": {"content": "<<</STUDENT_SHEET>>>"}})
            + "\n"
        ).encode("utf-8")

        captured: dict = {}

        def fake_urlopen(req, timeout=600.0):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(stream_lines)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = local_runner._generate_ollama(
                base_url="http://localhost:11434",
                model="qwen3.5-exam",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.6,
                max_tokens=512,
                stream=True,
            )
        self.assertEqual(captured["payload"]["think"], False)
        self.assertIn("<<<STUDENT_SHEET>>>", out)
        self.assertNotIn("<think>", out)

    def test_openai_backend_400_fallback_mocked(self) -> None:
        calls: list[dict] = []

        def fake_urlopen(req, timeout=600.0):
            payload = json.loads(req.data.decode("utf-8"))
            calls.append(payload)
            if "chat_template_kwargs" in payload:
                raise urllib.error.HTTPError(
                    req.full_url, 400, "Bad Request", {}, io.BytesIO(b"unsupported")
                )
            resp_body = json.dumps(
                {"choices": [{"message": {"content": "<<<STUDENT_SHEET>>>\nOK\n<<</STUDENT_SHEET>>>"}}]}
            ).encode("utf-8")
            return _FakeHTTPResponse(resp_body)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = local_runner._generate_openai(
                base_url="http://localhost:8080/v1",
                model="qwen3.5-exam",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.6,
                max_tokens=512,
            )
        self.assertEqual(len(calls), 2)
        self.assertIn("chat_template_kwargs", calls[0])
        self.assertNotIn("chat_template_kwargs", calls[1])
        self.assertIn("<<<STUDENT_SHEET>>>", out)

    def test_eval_val_mocked_backend(self) -> None:
        rec = _make_valid_record(level="SMA")
        with tempfile.TemporaryDirectory() as tmpdir:
            val_path = Path(tmpdir) / "val.jsonl"
            val_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            mock_assistant = rec["messages"][2]["content"]
            with patch("local_runner._generate_ollama", return_value=mock_assistant):
                with patch.object(
                    sys,
                    "argv",
                    ["local_runner.py", "--backend", "ollama", "--eval-val", str(val_path), "--limit", "1"],
                ):
                    rc = local_runner.main()
        self.assertEqual(rc, 0)


class TestNotebookAndHygiene(unittest.TestCase):
    def test_notebook_valid_and_configured(self) -> None:
        nb_path = ROOT / "FineTune_Qwen_Exam_Generator.ipynb"
        self.assertTrue(nb_path.exists())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)

        all_code_lines: list[str] = []
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            filtered = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(filtered, filename=f"cell_{idx}")
            all_code_lines.append(src)

        joined = "\n".join(all_code_lines)
        cells = nb.get("cells", [])
        setup_source = next(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "code"
            and "loaded_numpy_version" in "".join(cell.get("source", []))
        )
        package_check_source = next(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "code"
            and "=== Package Versions ===" in "".join(cell.get("source", []))
        )
        step5_source = next(
            "".join(cell.get("source", []))
            for cell in cells
            if cell.get("cell_type") == "code"
            and "def generate_exam(" in "".join(cell.get("source", []))
        )
        self.assertIn("initial_numpy_version", setup_source)
        self.assertIn("*numpy_pin", setup_source)
        self.assertIn("loaded_numpy_version != installed_numpy_version", setup_source)
        self.assertLess(package_check_source.index("import unsloth"), package_check_source.index("import torch"))
        self.assertIn("max_new_tokens: int = 3072", step5_source)
        self.assertIn("raw[-1800:]", step5_source)
        step6_index = next(
            idx
            for idx, cell in enumerate(cells)
            if cell.get("cell_type") == "code"
            and "model.save_pretrained_merged" in "".join(cell.get("source", []))
        )
        step6_source = "".join(cells[step6_index].get("source", []))
        prior_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells[:step6_index]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("def run_cmd(", step6_source)
        for prerequisite in (
            "model, tokenizer = FastLanguageModel.from_pretrained(",
            "response_part =",
            "CANONICAL_SYSTEM_PROMPT =",
        ):
            self.assertIn(prerequisite, prior_code)
        self.assertIn("unsloth/Qwen3.5-2B", joined)
        self.assertIn("load_in_4bit = False", joined)
        self.assertIn('"Q4_K_M"', joined)
        self.assertIn("save_pretrained_merged", joined)
        self.assertIn("convert_hf_to_gguf.py", joined)
        self.assertIn("llama-quantize", joined)
        self.assertIn("ssm_beta.weight", joined)
        self.assertNotIn("model.save_pretrained_gguf(", joined)
        self.assertIn("<<<STUDENT_SHEET>>>", joined)
        self.assertIn("<<<ANSWER_KEY>>>", joined)

    def test_workspace_hygiene_no_legacy_refs(self) -> None:
        forbidden = ("glm-4", "glm_client", "GLM_API", "zhipu", "bigmodel", "z.ai")
        exts = {".py", ".md", ".jsonl", ".json", ".ipynb", ".txt", ".template"}
        self_path = Path(__file__).resolve()
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.resolve() == self_path:
                continue
            if "__pycache__" in path.parts or ".git" in path.parts:
                continue
            if path.suffix.lower() not in exts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden:
                self.assertNotIn(
                    term.lower(),
                    text.lower(),
                    f"Found forbidden legacy term '{term}' in {path}",
                )


if __name__ == "__main__":
    unittest.main()
