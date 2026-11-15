"""Canonical dataset schema, prompt builder, and output parser for Qwen3.5-2B Exam Distillation.

This module is the single source of truth for:
- CANONICAL_SYSTEM_PROMPT (strictly faithful to .agents/skills/exam-generator/SKILL.md)
- Canonical enums/constants (EDUCATION_LEVELS, BLOOM_PROFILES, QUESTION_FORMATS)
- expected_mcq_option_count(education_level)
- build_user_prompt(...)
- strip_thinking(text) and parse_exam_output(text)
"""

from __future__ import annotations

import re
from typing import Sequence

STUDENT_SHEET_START = "<<<STUDENT_SHEET>>>"
STUDENT_SHEET_END = "<<</STUDENT_SHEET>>>"
ANSWER_KEY_START = "<<<ANSWER_KEY>>>"
ANSWER_KEY_END = "<<</ANSWER_KEY>>>"

CANONICAL_SYSTEM_PROMPT = """You are an expert exam designer. Create rigorous exams matching the requested topic, audience level, format, count, and language (Indonesian or English).

Rules:
- MCQ counts: 4 options (A-D) for SD/SMP; 5 options (A-E) for SMA/SMK/Kuliah/University.
- Stems & stimuli: Self-contained with all constants/data. Ground C3-C6 items in authentic stimuli (data, code, scenarios). Bold and capitalize negative markers (**BUKAN**, **KECUALI**, **NOT**, **EXCEPT**); no double negatives.
- Distractors: Misconception-driven, same-scope, length- and tone-calibrated. No "All/None of the above" or "Semua benar/salah". Order numeric/sequential options monotonically.
- Complex MCQ & Essay/Case: Complex MCQs need independently verifiable statements and an explicit scoring/selection rule. Essays/cases need concrete tasks (predict, trace, diagnose, fix, defend trade-offs) with per-part point weights (e.g., [10 poin] or [10 pts]).
- Audit: Solve from scratch, balance MCQ key letters, and exclude answers, hints, and Bloom labels from the student sheet unless requested.

Output ONLY these two tagged sections:
<<<STUDENT_SHEET>>>
Header, instructions, and numbered questions only.
<<</STUDENT_SHEET>>>
<<<ANSWER_KEY>>>
MCQ: correct letter, why right, and one line per distractor naming its misconception. Essay/Case: model solution and point breakdown summing to each sub-part weight.
<<</ANSWER_KEY>>>"""

EDUCATION_LEVELS: tuple[str, ...] = (
    "SD",
    "SMP",
    "SMA",
    "SMK",
    "Kuliah",
    "University",
)

BLOOM_PROFILES: tuple[str, ...] = (
    "C1-C2",
    "C3-C4",
    "C5-C6",
    "C4-C6",
    "C1-C6",
)

QUESTION_FORMATS: tuple[str, ...] = (
    "MCQ",
    "Complex MCQ",
    "Essay",
    "Case Study",
    "Mixed",
)

_LEVEL_FOUR_OPTIONS = {
    "sd",
    "sd/mi",
    "mi",
    "elementary",
    "primary",
    "smp",
    "smp/mts",
    "mts",
    "middle",
    "middle school",
    "junior high",
}

_LEVEL_FIVE_OPTIONS = {
    "sma",
    "sma/ma",
    "ma",
    "smk",
    "high school",
    "senior high",
    "vocational",
    "kuliah",
    "universitas",
    "university",
    "perguruan tinggi",
    "mahasiswa",
    "undergraduate",
    "graduate",
    "s1",
    "s2",
    "d3",
    "college",
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_TAG_STUDENT_RE = re.compile(
    r"<<<STUDENT_SHEET>>>(.*?)<<</STUDENT_SHEET>>>", re.DOTALL
)
_TAG_ANSWER_RE = re.compile(
    r"<<<ANSWER_KEY>>>(.*?)<<</ANSWER_KEY>>>", re.DOTALL
)

_FALLBACK_STUDENT_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,4}[ \t]*|\*{2})?(?:\d+\.[ \t]*)?"
    r"(?:Student[ \t]+Exam[ \t]+Sheet|Student[ \t]+Sheet|Lembar[ \t]+Soal(?:[ \t]+Siswa)?)"
    r"(?:\*{2})?[ \t]*:?[ \t]*$"
)
_FALLBACK_ANSWER_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,4}[ \t]*|\*{2})?(?:\d+\.[ \t]*)?"
    r"(?:Answer[ \t]+Key(?:[ \t]*(?:&|and)[ \t]*Rationale)?|"
    r"Kunci[ \t]+Jawaban(?:[ \t]*(?:&|dan)[ \t]*Pembahasan)?)"
    r"(?:\*{2})?[ \t]*:?[ \t]*$"
)


def expected_mcq_option_count(education_level: str) -> int:
    """Return required MCQ option count: 4 for SD/SMP, 5 for SMA/SMK/Kuliah/University."""
    if not isinstance(education_level, str) or not education_level.strip():
        raise ValueError("education_level must be a non-empty string")

    norm = re.sub(r"\s+", " ", education_level.strip().lower())
    if norm in _LEVEL_FOUR_OPTIONS:
        return 4
    if norm in _LEVEL_FIVE_OPTIONS:
        return 5

    # Substring fallback for composite labels like "SMA Kelas 11" or "Kuliah (S1)"
    tokens = set(re.findall(r"[a-z0-9]+", norm))
    if tokens & {"sd", "mi", "smp", "mts", "elementary", "primary"}:
        return 4
    if tokens & {
        "sma",
        "ma",
        "smk",
        "kuliah",
        "university",
        "universitas",
        "mahasiswa",
        "college",
        "undergraduate",
        "graduate",
        "s1",
        "s2",
        "d3",
    }:
        return 5

    raise ValueError(f"Unknown education_level: {education_level!r}")


def _normalize_language(language: str) -> str:
    norm = (language or "id").strip().lower()
    if norm in {"id", "ind", "indonesia", "indonesian", "bahasa indonesia"}:
        return "id"
    if norm in {"en", "eng", "english", "inggris", "bahasa inggris"}:
        return "en"
    raise ValueError(f"Unsupported language: {language!r} (expected 'id' or 'en')")


def _format_question_formats(question_formats: str | Sequence[str]) -> str:
    if isinstance(question_formats, str):
        cleaned = question_formats.strip()
        if not cleaned:
            raise ValueError("question_formats must not be empty")
        return cleaned
    parts = [str(fmt).strip() for fmt in question_formats if str(fmt).strip()]
    if not parts:
        raise ValueError("question_formats must not be empty")
    return ", ".join(parts)


def build_user_prompt(
    topic: str,
    education_level: str,
    bloom_profile: str,
    question_formats: str | Sequence[str],
    num_questions: int | str,
    language: str = "id",
    reference_material: str | None = None,
    custom_instructions: str | None = None,
    show_bloom_labels: bool = False,
) -> str:
    """Build a standardized user prompt for exam generation training and inference."""
    topic_clean = topic.strip()
    if not topic_clean:
        raise ValueError("topic must be non-empty")

    level_clean = education_level.strip()
    # Validate education_level early so invalid levels fail fast
    expected_mcq_option_count(level_clean)

    bloom_clean = bloom_profile.strip()
    if not bloom_clean:
        raise ValueError("bloom_profile must be non-empty")

    formats_str = _format_question_formats(question_formats)
    lang = _normalize_language(language)

    if lang == "id":
        lines = [
            "Buatkan paket soal ujian dengan spesifikasi berikut:",
            f"- Topik: {topic_clean}",
            f"- Jenjang Pendidikan: {level_clean}",
            f"- Profil Kognitif (Bloom): {bloom_clean}",
            f"- Format Soal: {formats_str}",
            f"- Jumlah Soal: {num_questions}",
            "- Bahasa: Bahasa Indonesia",
        ]
        if show_bloom_labels:
            lines.append(
                "- Label Bloom: Sertakan label level Bloom (mis. [C3]) pada setiap soal di Lembar Soal."
            )
        if reference_material and reference_material.strip():
            lines.append("")
            lines.append("Materi Referensi:")
            lines.append(reference_material.strip())
        if custom_instructions and custom_instructions.strip():
            lines.append("")
            lines.append("Instruksi Tambahan:")
            lines.append(custom_instructions.strip())
    else:
        lines = [
            "Create an exam package with the following specifications:",
            f"- Topic: {topic_clean}",
            f"- Education Level: {level_clean}",
            f"- Cognitive Profile (Bloom): {bloom_clean}",
            f"- Question Formats: {formats_str}",
            f"- Number of Questions: {num_questions}",
            "- Language: English",
        ]
        if show_bloom_labels:
            lines.append(
                "- Bloom Labels: Include Bloom taxonomy labels (e.g., [C3]) on each question in the Student Sheet."
            )
        if reference_material and reference_material.strip():
            lines.append("")
            lines.append("Reference Material:")
            lines.append(reference_material.strip())
        if custom_instructions and custom_instructions.strip():
            lines.append("")
            lines.append("Additional Instructions:")
            lines.append(custom_instructions.strip())

    return "\n".join(lines)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks if present."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Remove any stray closing/opening think tag if malformed
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def parse_exam_output(text: str) -> dict[str, str]:
    """Extract student_sheet, answer_key, and raw_clean from model output.

    Supports canonical <<<STUDENT_SHEET>>> / <<<ANSWER_KEY>>> tags first,
    with fallback header parsing for Student Exam Sheet / Lembar Soal Siswa
    and Answer Key / Kunci Jawaban.
    """
    raw_clean = strip_thinking(text)
    if not raw_clean:
        return {"student_sheet": "", "answer_key": "", "raw_clean": ""}

    student_match = _TAG_STUDENT_RE.search(raw_clean)
    answer_match = _TAG_ANSWER_RE.search(raw_clean)

    if student_match or answer_match:
        student_sheet = student_match.group(1).strip() if student_match else ""
        answer_key = answer_match.group(1).strip() if answer_match else ""
        return {
            "student_sheet": student_sheet,
            "answer_key": answer_key,
            "raw_clean": raw_clean,
        }

    # Fallback header parsing
    ans_header = _FALLBACK_ANSWER_HEADER_RE.search(raw_clean)
    stu_header = _FALLBACK_STUDENT_HEADER_RE.search(raw_clean)

    student_sheet = ""
    answer_key = ""

    if stu_header and ans_header and stu_header.start() < ans_header.start():
        student_sheet = raw_clean[stu_header.end() : ans_header.start()].strip()
        answer_key = raw_clean[ans_header.end() :].strip()
    elif ans_header:
        student_sheet = raw_clean[: ans_header.start()].strip()
        if stu_header and stu_header.start() < ans_header.start():
            student_sheet = raw_clean[stu_header.end() : ans_header.start()].strip()
        answer_key = raw_clean[ans_header.end() :].strip()
    elif stu_header:
        student_sheet = raw_clean[stu_header.end() :].strip()

    return {
        "student_sheet": student_sheet,
        "answer_key": answer_key,
        "raw_clean": raw_clean,
    }


__all__ = [
    "ANSWER_KEY_END",
    "ANSWER_KEY_START",
    "BLOOM_PROFILES",
    "CANONICAL_SYSTEM_PROMPT",
    "EDUCATION_LEVELS",
    "QUESTION_FORMATS",
    "STUDENT_SHEET_END",
    "STUDENT_SHEET_START",
    "build_user_prompt",
    "expected_mcq_option_count",
    "parse_exam_output",
    "strip_thinking",
]
