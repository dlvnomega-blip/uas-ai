---
name: exam-generator
description: >-
  Use when creating exams, quizzes, or practice questions from a topic, syllabus, or
  study material. Enforces misconception-first item design, same-scope distractors,
  calibrated options, solve-from-scratch verification, and clean student/key sheets.
---

# Exam Generator

Build exam questions around what separates genuine mechanistic understanding from surface memorization or test-taking intuition. Match the user's topic, audience level, and language.

## Workflow

### 1. Scope & Trap-First Planning
1. Confirm **topic/source**, **audience level** (`4` MCQ options `A`–`D` for `SD`/`SMP`; `5` options `A`–`E` for `SMA`/University), **formats & counts**, and **language**.
2. Extract distinct **concepts, edge cases, and failure modes** from the source so no two items test the same idea. Plan coverage across Recall/Understand (`C1`–`C2`), Apply/Analyze (`C3`–`C4`), and Evaluate/Create (`C5`–`C6`) based on what the examinee must *do*, not topic jargon.
3. **Avoid pure regurgitation essays** (e.g., *"Compare 4 textbook traits of X vs Y"*) unless recall drills are requested. Prefer prompts that force examinees to **predict outcomes, trace steps, diagnose failures, fix flaws, or defend trade-offs**.

### 2. Write Tight, Self-Contained Stems & Stimuli
- **Authentic evidence (`C3`–`C6`)**: Anchor applied and analytical items in concrete stimuli (data, sequences, code, experimental results, constraints) that must actively be used—never decorative story fluff around a definition.
- **Stem–key contract**: Ensure the stem asks for what the options actually deliver (never ask to *"evaluate severity"* if the options only classify definitions).
- **Self-contained data**: Include every constant, table subset, or rule needed across all sub-parts of an item (including follow-up parts) unless memorization is explicitly expected at that level.
- **Clean & consistent framing**: One central problem per stem, answerable before reading options. Maintain strict organism/domain consistency across stem and all options. Prefer positive phrasing; if negative (`NOT`, `EXCEPT`, `BUKAN`, `KECUALI`) is unavoidable, **BOLD AND CAPITALIZE** it (never double negatives). Remove keyword echoes between stem and key.

### 3. Engineer Same-Scope, Calibrated Distractors
- **Multiple Choice (Single Best Answer)**:
  1. Write the single unambiguously best answer first.
  2. **Same-Scope Rule**: Every distractor must operate inside the **exact same stage, mechanism, compartment, or category** left open by the stem—never pad with out-of-scope options that test-wise examinees eliminate without knowing the concept.
  3. **No Downstream-Truth Ambiguity**: Reject any distractor that could be defended as a secondary consequence of the right answer. Every distractor must be categorically false under the stimulus.
  4. **Misconception-Driven**: Derive wrong options from real errors—computed wrong calculation paths, swapped neighboring mechanisms, reversed causality, or plausible-sounding false rules. Never use *"All/None of the above"*.
  5. **Calibrate Option Weight**: Match word count, clause complexity, specificity, and tone across all options—never let the key stand out as the longest, most nuanced, or most "textbook-complete" option. Sort numeric/ordered options monotonically.
- **Complex MCQ** (when requested): Make each statement independently verifiable and state the exact selection/scoring rule.
- **Essay / Case**: Bound deliverables with specific tasks and point weights per sub-part (e.g., `[10 pts]`).

### 4. Solve-From-Scratch Audit & Delivery
Before outputting, silently verify:
1. **Solve every item from scratch** using *only* the information provided in the exam (check math, indices, polarity, units, and that every required lookup value is present).
2. **Balance key letters**: Count correct letters across all MCQs (`A`–`D` or `A`–`E`); swap option order if any letter appears `0` times or dominates.
3. **Fact-check rationales**: Ensure explanations contain no imprecise generalizations or context mismatches.
4. **Zero leaks**: Confirm the Student Exam Sheet contains no answers, hints, or Bloom labels (unless requested).

Present two cleanly separated sections:
1. **Student Exam Sheet** — Header, instructions, and numbered questions only.
2. **Answer Key & Rationale** —
   - **MCQ**: Correct letter, why it is right, and one line per distractor stating the specific misconception/error it represents.
   - **Essay / Case**: Model solution and concise point breakdown summing to each sub-part's weight.
