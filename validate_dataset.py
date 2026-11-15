"""Dataset validator and CLI for Qwen3.5-2B Exam Distillation JSONL files.

Usage:
    python validate_dataset.py <file1.jsonl> [file2.jsonl ...]

Importable API:
    validate_record(record, idx=0) -> (errors, warnings, stats)
    validate_jsonl(path) -> (errors, warnings, stats)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from schema import (
    ANSWER_KEY_END,
    ANSWER_KEY_START,
    CANONICAL_SYSTEM_PROMPT,
    STUDENT_SHEET_END,
    STUDENT_SHEET_START,
    expected_mcq_option_count,
)

MAX_RECORD_CHARS = 11500

_THINK_TAG_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_STUDENT_BLOCK_RE = re.compile(
    rf"{re.escape(STUDENT_SHEET_START)}(.*?){re.escape(STUDENT_SHEET_END)}",
    re.DOTALL,
)
_ANSWER_BLOCK_RE = re.compile(
    rf"{re.escape(ANSWER_KEY_START)}(.*?){re.escape(ANSWER_KEY_END)}",
    re.DOTALL,
)

_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Kunci:", re.compile(r"(?i)\bkunci\s*:")),
    ("Jawaban Benar:", re.compile(r"(?i)\bjawaban\s+benar\s*:")),
    ("Answer Key", re.compile(r"(?i)\banswer\s+key\b")),
    ("Miskonsepsi", re.compile(r"(?i)\bmiskonsepsi\b")),
    ("Pembahasan:", re.compile(r"(?i)\bpembahasan\s*:")),
)

# Match Bloom tags like (C1), (C2-C3), [C4], (C5: Evaluate), Level Bloom
# while avoiding false positives on chemical formulas such as (C2H5OH) or (C6H12O6).
_BLOOM_TAG_RE = re.compile(
    r"(?i)(?:\(|\[)\s*C[1-6](?=(?:\)|\]|:|-|–|/|,|\s))|\blevel\s+bloom\b"
)

_FORBIDDEN_OPTION_RE = re.compile(
    r"(?i)\b(?:"
    r"semua\s+jawaban\s+benar|"
    r"semua\s+di\s+atas\s+benar|"
    r"semua\s+pilihan\s+benar|"
    r"tidak\s+ada\s+yang\s+benar|"
    r"tidak\s+ada\s+jawaban\s+yang\s+benar|"
    r"all\s+of\s+the\s+above|"
    r"none\s+of\s+the\s+above"
    r")\b"
)

_OPTION_LINE_RE = re.compile(
    r"^[ \t]*(?:\*\*)?([A-E])[.)](?:\*\*)?[ \t]+(.+?)\s*$"
)
_QUESTION_START_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,4}[ \t]*)?(?:\*\*)?(?:(?:Soal|Question|No\.?|Item)[ \t]+)?(\d+)[.)](?:\*\*)?[ \t]+"
)
_POINT_WEIGHT_RE = re.compile(
    r"(?i)(?:\[|\()\s*\d+(?:[.,]\d+)?\s*(?:poin|pts?|points?|skor|bobot)\s*(?:\]|\))"
)

_COMPLEX_STEM_RE = re.compile(
    r"(?i)(?:"
    r"pilih(?:lah)?\s+lebih\s+dari\s+satu|"
    r"pilih\s+semua\s+(?:jawaban\s+)?yang\s+benar|"
    r"kompleks|complex\s+mcq|"
    r"pernyataan\s+(?:berikut|di\s+bawah)|"
    r"\(\s*1\s*\)\s*,?\s*\(\s*2\s*\)|"
    r"\bI\b.*\bII\b.*\bIII\b|"
    r"benar\s*/\s*salah|true\s*/\s*false|"
    r"centang|check\s+all"
    r")"
)

_ANSWER_KEY_LETTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # e.g. "1. **Kunci: C**", "- Jawaban Benar: B", "**Answer:** D"
    re.compile(
        r"(?im)^[ \t]*(?:(?:#{1,4}[ \t]*)?(?:(?:Soal|Question|No\.?|Item)[ \t]*)?\d+[.)][ \t]*)?"
        r"(?:[-*][ \t]*)?(?:\*{0,2})(?:Kunci(?:\s+Jawaban)?|Jawaban(?:\s+Benar)?|"
        r"Correct(?:\s+Answer)?|Answer(?:\s+Key)?|Key)(?:\*{0,2})[ \t]*[:\-][ \t]*"
        r"(?:\*{0,2})([A-E])\b(?:\*{0,2})(?![A-Za-z0-9]|[, \t]+(?:dan|and|&)[ \t]*[A-E]\b|,[ \t]*[A-E]\b)"
    ),
    # e.g. "1. **C** — ..." or "1. C."
    re.compile(
        r"(?im)^[ \t]*(?:#{1,4}[ \t]*)?(?:(?:Soal|Question|No\.?)[ \t]*)?\d+[.)][ \t]*"
        r"(?:\*{0,2})([A-E])\b(?:\*{0,2})(?=[ \t]*(?:$|[-–—:(.\[]))"
    ),
)

_MCQ_RATIONALE_RE = re.compile(
    r"(?i)(?:"
    r"miskonsepsi|misconception|mengapa\s+benar|alasan|why\s+(?:it\s+is\s+)?right|"
    r"why\s+correct|rationale|pengecoh|distractor|salah\s+karena|keliru|incorrect"
    r")"
)
_DISTRACTOR_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:\*{0,2})[A-E](?:\*{0,2})\s*(?:[:.\-)]|\([^)]*\)\s*:)"
)

_ESSAY_BREAKDOWN_RE = re.compile(
    r"(?i)(?:"
    r"\d+(?:[.,]\d+)?\s*(?:poin|pts?|points?|skor)|"
    r"rubrik|rubric|breakdown|pembagian\s+poin|rincian\s+(?:skor|poin|nilai)|"
    r"total\s*(?::|=|\s+\d+)"
    r")"
)

_UNBOLDED_NEG_RE = re.compile(r"(?<!\*\*)\b(BUKAN|KECUALI|NOT|EXCEPT)\b(?!\*\*)")


def _extract_level_from_user(user_prompt: str) -> tuple[str | None, int | None]:
    """Infer education level and expected MCQ option count from the user prompt."""
    level_match = re.search(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:Jenjang\s+Pendidikan|Education\s+Level|Audience\s+Level|Level|Jenjang)\s*:\s*(.+?)\s*$",
        user_prompt,
    )
    candidates: list[str] = []
    if level_match:
        candidates.append(level_match.group(1).strip())
    candidates.append(user_prompt)

    canonical_map = [
        ("SD", re.compile(r"(?i)\b(?:SD(?:/MI)?|elementary|primary)\b")),
        ("SMP", re.compile(r"(?i)\b(?:SMP(?:/MTs)?|middle\s+school|junior\s+high)\b")),
        ("SMK", re.compile(r"(?i)\bSMK\b")),
        ("SMA", re.compile(r"(?i)\b(?:SMA(?:/MA)?|high\s+school|senior\s+high)\b")),
        (
            "Kuliah",
            re.compile(r"(?i)\b(?:kuliah|perguruan\s+tinggi|mahasiswa|universitas)\b"),
        ),
        (
            "University",
            re.compile(r"(?i)\b(?:university|college|undergraduate|graduate)\b"),
        ),
    ]

    for text in candidates:
        for canonical, pat in canonical_map:
            if pat.search(text):
                return canonical, expected_mcq_option_count(canonical)
        if level_match and text == candidates[0]:
            try:
                return text, expected_mcq_option_count(text)
            except ValueError:
                pass

    return None, None


def _extract_language_from_user(user_prompt: str) -> str:
    lang_match = re.search(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:Bahasa|Language)\s*:\s*(.+?)\s*$",
        user_prompt,
    )
    if lang_match:
        val = lang_match.group(1).strip().lower()
        if any(k in val for k in ("english", "inggris", "en")):
            return "en"
        if any(k in val for k in ("indonesia", "indonesian", "id")):
            return "id"

    en_hits = len(
        re.findall(
            r"(?i)\b(?:create|exam|topic|education\s+level|question|number\s+of)\b",
            user_prompt,
        )
    )
    id_hits = len(
        re.findall(
            r"(?i)\b(?:buatkan|soal|ujian|topik|jenjang|jumlah\s+soal|bahasa)\b",
            user_prompt,
        )
    )
    return "en" if en_hits > id_hits else "id"


def _user_requests_bloom_labels(user_prompt: str) -> bool:
    """Return True only when the user prompt affirmatively asks to show Bloom labels."""
    if re.search(
        r"(?i)(?:jangan|tidak|tanpa|sembunyikan|do\s+not|don't|without|exclude|hide)"
        r"[^\n.]{0,40}bloom",
        user_prompt,
    ):
        return False
    return bool(
        re.search(
            r"(?i)(?:"
            r"(?:tampilkan|sertakan|munculkan|cantumkan|berikan|include|show|display|with)"
            r"[^\n.]{0,40}(?:(?:label|tag|level)\s*(?:taksonomi\s*)?bloom|bloom\s*(?:taxonomy\s*)?(?:labels?|tags?|levels?))|"
            r"(?:label|bloom)\s*(?:bloom|labels?)\s*:\s*(?:ya|yes|tampilkan|sertakan|include|show|true)"
            r")",
            user_prompt,
        )
    )


def _extract_formats_from_user(user_prompt: str) -> dict[str, bool]:
    fmt_match = re.search(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:Format\s+Soal|Question\s+Formats?|Format)\s*:\s*(.+?)\s*$",
        user_prompt,
    )
    text = fmt_match.group(1) if fmt_match else user_prompt
    has_complex = bool(
        re.search(r"(?i)\b(?:complex\s+mcq|pg\s+kompleks|pilihan\s+ganda\s+kompleks)\b", text)
    )
    has_std_mcq = bool(
        re.search(r"(?i)\b(?:mcq|pilihan\s+ganda|single\s+best|pg\s+tunggal)\b", text)
    ) and not (has_complex and not re.search(r"(?i)\bmixed\b|\bcampuran\b|,", text))
    has_essay = bool(
        re.search(r"(?i)\b(?:essay|uraian|case\s*study|studi\s+kasus|esai)\b", text)
    )
    has_mixed = bool(re.search(r"(?i)\b(?:mixed|campuran)\b", text))
    return {
        "has_complex": has_complex,
        "has_std_mcq": has_std_mcq or has_mixed,
        "has_essay": has_essay or has_mixed,
    }


def _parse_mcq_blocks(student_sheet: str) -> list[dict[str, Any]]:
    """Detect MCQ option blocks (starting at A.) along with preceding stem text."""
    lines = student_sheet.splitlines()
    blocks: list[dict[str, Any]] = []
    stem_lines: list[str] = []
    current_block: dict[str, Any] | None = None

    def flush_block() -> None:
        nonlocal current_block
        if current_block is not None:
            blocks.append(current_block)
            current_block = None

    for line in lines:
        opt_m = _OPTION_LINE_RE.match(line)
        if opt_m:
            letter = opt_m.group(1).upper()
            opt_text = opt_m.group(2).strip()
            if letter == "A":
                flush_block()
                current_block = {
                    "stem": "\n".join(stem_lines).strip(),
                    "options": [(letter, opt_text)],
                }
                stem_lines = []
                continue
            if current_block is not None:
                prev_letters = [l for l, _ in current_block["options"]]
                if letter not in prev_letters:
                    current_block["options"].append((letter, opt_text))
                    continue
        if current_block is not None and not line.strip():
            flush_block()
        elif current_block is not None and _QUESTION_START_RE.match(line):
            flush_block()

        if current_block is None:
            if _QUESTION_START_RE.match(line):
                stem_lines = [line]
            else:
                stem_lines.append(line)

    flush_block()
    return blocks


def _extract_answer_key_letters(answer_key: str) -> list[str]:
    """Extract ordered MCQ answer letters (A-E) from <<<ANSWER_KEY>>>."""
    letters_by_line: list[tuple[int, str]] = []
    seen_lines: set[int] = set()
    lines = answer_key.splitlines()

    for idx, line in enumerate(lines):
        for pat in _ANSWER_KEY_LETTER_PATTERNS:
            m = pat.match(line)
            if m:
                letters_by_line.append((idx, m.group(1).upper()))
                seen_lines.add(idx)
                break

    # Fallback: check inline patterns like "1. **Kunci: B**" anywhere on a line not yet matched
    if not letters_by_line:
        inline_re = re.compile(
            r"(?i)(?:Kunci(?:\s+Jawaban)?|Jawaban(?:\s+Benar)?|"
            r"Correct(?:\s+Answer)?|Answer(?:\s+Key)?|Key)\s*[:\-]\s*(?:\*{0,2})([A-E])\b"
        )
        for idx, line in enumerate(lines):
            if idx in seen_lines:
                continue
            m = inline_re.search(line)
            if m:
                letters_by_line.append((idx, m.group(1).upper()))

    return [letter for _, letter in sorted(letters_by_line, key=lambda x: x[0])]


def validate_record(
    record: Any, idx: int = 0
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate a single JSONL record against the Qwen3.5-2B Exam Distillation contract.

    Returns (errors, warnings, stats).
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "index": idx,
        "char_len": 0,
        "language": "unknown",
        "education_level": "unknown",
        "expected_mcq_options": None,
        "mcq_count": 0,
        "complex_mcq_count": 0,
        "essay_count": 0,
        "key_letters": [],
        "user_prompt_norm": "",
    }

    prefix = f"[record {idx}] "

    if not isinstance(record, dict):
        errors.append(f"{prefix}Record must be a JSON object.")
        return errors, warnings, stats

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(
            f"{prefix}'messages' must be a list of exactly 3 turns (system, user, assistant)."
        )
        return errors, warnings, stats

    expected_roles = ("system", "user", "assistant")
    contents: list[str] = []
    for turn_idx, expected_role in enumerate(expected_roles):
        turn = messages[turn_idx]
        if not isinstance(turn, dict):
            errors.append(f"{prefix}messages[{turn_idx}] must be an object.")
            return errors, warnings, stats
        role = turn.get("role")
        content = turn.get("content")
        if role != expected_role:
            errors.append(
                f"{prefix}messages[{turn_idx}].role must be '{expected_role}', got {role!r}."
            )
        if not isinstance(content, str):
            errors.append(
                f"{prefix}messages[{turn_idx}].content must be a string."
            )
            return errors, warnings, stats
        contents.append(content)

    system_msg, user_msg, assistant_msg = contents
    total_chars = len(system_msg) + len(user_msg) + len(assistant_msg)
    stats["char_len"] = total_chars
    stats["user_prompt_norm"] = user_msg.strip()

    if system_msg != CANONICAL_SYSTEM_PROMPT:
        errors.append(
            f"{prefix}System message does not match CANONICAL_SYSTEM_PROMPT exactly."
        )

    if not user_msg.strip():
        errors.append(f"{prefix}User prompt must be non-empty.")

    if not assistant_msg.strip():
        errors.append(f"{prefix}Assistant output must be non-empty.")
        return errors, warnings, stats

    if _THINK_TAG_RE.search(assistant_msg):
        errors.append(
            f"{prefix}Assistant output must not contain <think> or </think> tags."
        )

    if total_chars > MAX_RECORD_CHARS:
        errors.append(
            f"{prefix}Total character length {total_chars} exceeds limit of {MAX_RECORD_CHARS}."
        )

    # Check exact tag structure
    stu_matches = list(_STUDENT_BLOCK_RE.finditer(assistant_msg))
    ans_matches = list(_ANSWER_BLOCK_RE.finditer(assistant_msg))

    if len(stu_matches) != 1 or len(ans_matches) != 1:
        errors.append(
            f"{prefix}Assistant output must contain exactly one "
            f"{STUDENT_SHEET_START}...{STUDENT_SHEET_END} block and one "
            f"{ANSWER_KEY_START}...{ANSWER_KEY_END} block."
        )
        return errors, warnings, stats

    stu_match = stu_matches[0]
    ans_match = ans_matches[0]

    if stu_match.end() > ans_match.start():
        errors.append(
            f"{prefix}{STUDENT_SHEET_START} block must precede {ANSWER_KEY_START} block."
        )

    outside_text = (
        assistant_msg[: stu_match.start()]
        + assistant_msg[stu_match.end() : ans_match.start()]
        + assistant_msg[ans_match.end() :]
    ).strip()
    if outside_text:
        errors.append(
            f"{prefix}Assistant output contains unexpected text outside the two canonical tag blocks."
        )

    student_sheet = stu_match.group(1).strip()
    answer_key = ans_match.group(1).strip()

    if not student_sheet:
        errors.append(f"{prefix}<<<STUDENT_SHEET>>> section is empty.")
    if not answer_key:
        errors.append(f"{prefix}<<<ANSWER_KEY>>> section is empty.")
    if not student_sheet or not answer_key:
        return errors, warnings, stats

    # Extract user specifications
    level_label, expected_opts = _extract_level_from_user(user_msg)
    lang = _extract_language_from_user(user_msg)
    formats_info = _extract_formats_from_user(user_msg)
    allow_bloom = _user_requests_bloom_labels(user_msg)

    stats["language"] = lang
    stats["education_level"] = level_label or "unknown"
    stats["expected_mcq_options"] = expected_opts

    if expected_opts is None:
        warnings.append(
            f"{prefix}Could not infer education level from user prompt for MCQ option count check."
        )

    # Zero-leak checks in student_sheet
    for leak_name, leak_re in _LEAK_PATTERNS:
        if leak_re.search(student_sheet):
            errors.append(
                f"{prefix}Answer/rationale leak detected in student_sheet ('{leak_name}')."
            )

    if not allow_bloom:
        bloom_hit = _BLOOM_TAG_RE.search(student_sheet)
        if bloom_hit:
            errors.append(
                f"{prefix}Bloom label leak detected in student_sheet ({bloom_hit.group(0)!r}) "
                f"when user prompt did not request Bloom labels."
            )

    # Non-blocking formatting warning for negative markers
    if _UNBOLDED_NEG_RE.search(student_sheet):
        warnings.append(
            f"{prefix}Negative marker (BUKAN/KECUALI/NOT/EXCEPT) in student_sheet should be **BOLD AND CAPITALIZED**."
        )

    # MCQ option blocks & forbidden options
    mcq_blocks = _parse_mcq_blocks(student_sheet)
    standard_mcq_count = 0
    complex_mcq_count = 0

    for block_idx, block in enumerate(mcq_blocks, start=1):
        letters = [letter for letter, _ in block["options"]]
        stem = block["stem"]

        for letter, opt_text in block["options"]:
            forb = _FORBIDDEN_OPTION_RE.search(opt_text)
            if forb:
                errors.append(
                    f"{prefix}Forbidden option phrase {forb.group(0)!r} in MCQ #{block_idx} option {letter}."
                )

        is_complex_block = bool(_COMPLEX_STEM_RE.search(stem)) or (
            formats_info["has_complex"] and not formats_info["has_std_mcq"]
        )

        if is_complex_block:
            complex_mcq_count += 1
        else:
            standard_mcq_count += 1

        if expected_opts == 4:
            if "E" in letters:
                errors.append(
                    f"{prefix}SD/SMP MCQ #{block_idx} has option E, but expected 4 options (A-D)."
                )
            elif not is_complex_block and letters != ["A", "B", "C", "D"]:
                errors.append(
                    f"{prefix}SD/SMP MCQ #{block_idx} has options {letters}, expected ['A', 'B', 'C', 'D']."
                )
        elif expected_opts == 5:
            if not is_complex_block:
                if "E" not in letters or letters != ["A", "B", "C", "D", "E"]:
                    errors.append(
                        f"{prefix}{level_label or 'SMA/Kuliah'} standard MCQ #{block_idx} "
                        f"has options {letters}, expected 5 options ['A', 'B', 'C', 'D', 'E']."
                    )

        # Option length calibration warning
        word_counts = [max(1, len(opt.split())) for _, opt in block["options"]]
        if len(word_counts) >= 4:
            sorted_wc = sorted(word_counts)
            median_wc = sorted_wc[len(sorted_wc) // 2]
            if max(word_counts) > max(12, median_wc * 3):
                warnings.append(
                    f"{prefix}MCQ #{block_idx} has poorly calibrated option lengths (word counts={word_counts})."
                )

    stats["mcq_count"] = standard_mcq_count
    stats["complex_mcq_count"] = complex_mcq_count

    # Detect essay / case study items
    question_numbers = _QUESTION_START_RE.findall(student_sheet)
    point_tags = _POINT_WEIGHT_RE.findall(student_sheet)
    total_numbered = len(question_numbers)
    inferred_essay_count = max(0, total_numbered - len(mcq_blocks))
    has_essay_items = (
        bool(point_tags)
        or (formats_info["has_essay"] and (inferred_essay_count > 0 or len(mcq_blocks) == 0))
        or (inferred_essay_count > 0 and complex_mcq_count == 0)
    )
    stats["essay_count"] = (
        inferred_essay_count if inferred_essay_count > 0 else (1 if has_essay_items else 0)
    )

    # Answer key checks for MCQs
    key_letters = _extract_answer_key_letters(answer_key)
    stats["key_letters"] = key_letters

    if standard_mcq_count > 0:
        if not key_letters:
            errors.append(
                f"{prefix}Student sheet has {standard_mcq_count} standard MCQ(s), "
                f"but no MCQ answer key letters were found in <<<ANSWER_KEY>>>."
            )
        else:
            if expected_opts == 4 and "E" in key_letters:
                errors.append(
                    f"{prefix}SD/SMP answer key contains letter 'E', outside A-D."
                )
            if len(key_letters) < standard_mcq_count:
                warnings.append(
                    f"{prefix}Extracted {len(key_letters)} MCQ key letter(s) for "
                    f"{standard_mcq_count} standard MCQ block(s)."
                )
            if len(key_letters) >= 5 and expected_opts in (4, 5):
                allowed = ("A", "B", "C", "D") if expected_opts == 4 else ("A", "B", "C", "D", "E")
                dist = Counter(key_letters)
                missing_letters = [ch for ch in allowed if dist[ch] == 0]
                max_letter, max_cnt = dist.most_common(1)[0]
                if missing_letters or max_cnt > len(key_letters) * 0.5:
                    warnings.append(
                        f"{prefix}MCQ key distribution is imbalanced ({dict(dist)})."
                    )

        has_rationale = bool(_MCQ_RATIONALE_RE.search(answer_key))
        distractor_lines = _DISTRACTOR_LINE_RE.findall(answer_key)
        if not has_rationale and len(distractor_lines) < max(2, standard_mcq_count):
            errors.append(
                f"{prefix}<<<ANSWER_KEY>>> lacks MCQ rationales/misconception explanations for distractors."
            )

    # Answer key checks for Essays / Case Studies
    if has_essay_items:
        if not point_tags and not _COMPLEX_STEM_RE.search(student_sheet):
            warnings.append(
                f"{prefix}Essay/Case items detected in student_sheet without explicit point weights (e.g., [10 poin] or [10 pts])."
            )
        if not _ESSAY_BREAKDOWN_RE.search(answer_key):
            errors.append(
                f"{prefix}Essay/Case items present, but <<<ANSWER_KEY>>> lacks point totals or point breakdown."
            )

    return errors, warnings, stats


def validate_jsonl(
    path: str | Path,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate every record in a JSONL file and aggregate file-level stats.

    Returns (errors, warnings, stats).
    """
    file_path = Path(path)
    all_errors: list[str] = []
    all_warnings: list[str] = []

    key_dist: Counter[str] = Counter({k: 0 for k in ("A", "B", "C", "D", "E")})
    lang_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    char_lengths: list[int] = []

    stats: dict[str, Any] = {
        "file": str(file_path),
        "total_records": 0,
        "valid_records": 0,
        "error_count": 0,
        "warning_count": 0,
        "duplicate_user_prompts": 0,
        "key_letter_distribution": dict(key_dist),
        "language_counts": {},
        "level_counts": {},
        "mcq_count": 0,
        "complex_mcq_count": 0,
        "essay_count": 0,
        "char_len_min": 0,
        "char_len_max": 0,
        "char_len_avg": 0.0,
    }

    if not file_path.exists():
        all_errors.append(f"File not found: {file_path}")
        stats["error_count"] = len(all_errors)
        return all_errors, all_warnings, stats

    seen_prompts: dict[str, int] = {}

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            stats["total_records"] += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                all_errors.append(f"[line {line_no}] Invalid JSON: {exc}")
                continue

            rec_errors, rec_warnings, rec_stats = validate_record(record, idx=line_no)

            prompt_norm = rec_stats.get("user_prompt_norm", "")
            if prompt_norm:
                if prompt_norm in seen_prompts:
                    first_line = seen_prompts[prompt_norm]
                    rec_errors.append(
                        f"[record {line_no}] Duplicate user prompt (first seen at line {first_line})."
                    )
                    stats["duplicate_user_prompts"] += 1
                else:
                    seen_prompts[prompt_norm] = line_no

            all_errors.extend(rec_errors)
            all_warnings.extend(rec_warnings)

            if not rec_errors:
                stats["valid_records"] += 1

            for letter in rec_stats.get("key_letters", []):
                if letter in key_dist:
                    key_dist[letter] += 1

            lang_counts[rec_stats.get("language", "unknown")] += 1
            level_counts[rec_stats.get("education_level", "unknown")] += 1
            stats["mcq_count"] += rec_stats.get("mcq_count", 0)
            stats["complex_mcq_count"] += rec_stats.get("complex_mcq_count", 0)
            stats["essay_count"] += rec_stats.get("essay_count", 0)

            char_len = rec_stats.get("char_len", 0)
            if char_len > 0:
                char_lengths.append(char_len)

    if stats["total_records"] == 0:
        all_errors.append(f"File contains 0 JSONL records: {file_path}")

    stats["error_count"] = len(all_errors)
    stats["warning_count"] = len(all_warnings)
    stats["key_letter_distribution"] = dict(key_dist)
    stats["language_counts"] = dict(lang_counts)
    stats["level_counts"] = dict(level_counts)
    if char_lengths:
        stats["char_len_min"] = min(char_lengths)
        stats["char_len_max"] = max(char_lengths)
        stats["char_len_avg"] = round(sum(char_lengths) / len(char_lengths), 1)

    return all_errors, all_warnings, stats


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "Usage: python validate_dataset.py <file1.jsonl> [file2.jsonl ...]",
            file=sys.stderr,
        )
        return 2

    overall_errors = 0
    for file_arg in args:
        errors, warnings, stats = validate_jsonl(file_arg)
        overall_errors += len(errors)

        status = "PASS" if not errors else "FAIL"
        print(f"=== {status}: {stats['file']} ===")
        print(
            f"Records: {stats['valid_records']}/{stats['total_records']} valid | "
            f"Errors: {stats['error_count']} | Warnings: {stats['warning_count']} | "
            f"Duplicates: {stats['duplicate_user_prompts']}"
        )
        print(
            f"Items: MCQ={stats['mcq_count']}, ComplexMCQ={stats['complex_mcq_count']}, "
            f"Essay={stats['essay_count']} | KeyDist={stats['key_letter_distribution']}"
        )
        print(
            f"Languages={stats['language_counts']} | Levels={stats['level_counts']} | "
            f"Chars(min/avg/max)={stats['char_len_min']}/{stats['char_len_avg']}/{stats['char_len_max']}"
        )

        for w in warnings[:20]:
            print(f"  WARN: {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more warning(s)")

        for e in errors[:30]:
            print(f"  ERR:  {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more error(s)")

    return 0 if overall_errors == 0 else 1


__all__ = [
    "MAX_RECORD_CHARS",
    "main",
    "validate_jsonl",
    "validate_record",
]

if __name__ == "__main__":
    raise SystemExit(main())
