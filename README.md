# Qwen3.5-2B Exam Generator Distillation Pipeline

End-to-end distillation pipeline for fine-tuning **Qwen3.5-2B** (`unsloth/Qwen3.5-2B`) into a specialized bilingual (Indonesian & English) exam generator following `.agents/skills/exam-generator/SKILL.md`.

The distilled model produces two tagged sections per request:
- `<<<STUDENT_SHEET>>>` ... `<<</STUDENT_SHEET>>>`: clean header, instructions, and numbered questions (4 options `A`–`D` for SD/SMP; 5 options `A`–`E` for SMA/SMK/Kuliah/University).
- `<<<ANSWER_KEY>>>` ... `<<</ANSWER_KEY>>>`: solved-from-scratch answer key, misconception-based distractor rationales, and point-breakdown rubrics for essays/case studies.

---

## Project Structure

```text
.
├── .agents/skills/exam-generator/SKILL.md  # Exam design specification
├── data_batches/                           # 8 stratified JSONL shards (12 records each)
├── schema.py                               # Canonical system prompt, prompt builder & output parser
├── prepare_dataset.py                      # Merges, validates, and splits shards (train/val)
├── validate_dataset.py                     # Structural, leak, Bloom, and option validator CLI
├── dataset_exam_generator.jsonl            # Merged dataset (96 records)
├── dataset_train.jsonl                     # Stratified training split (86 records)
├── dataset_val.jsonl                       # Stratified validation split (10 records)
├── FineTune_Qwen_Exam_Generator.ipynb      # Google Colab T4 16-bit LoRA + GGUF Q4_K_M notebook
├── Modelfile.template                      # Ollama Modelfile template for Q4_K_M GGUF
├── local_runner.py                         # Multi-backend local inference & eval CLI
├── requirements.txt                        # Python dependencies for local HF/PDF support
└── tests/
    └── test_distillation_pipeline.py       # Unit & hygiene test suite
```

---

## Step 1: Dataset Preparation & Validation

Install local dependencies (only `pypdf` is needed for PDF reference files; `transformers>=5.2.0` is needed for `--backend hf`):

```bash
pip install -r requirements.txt
```

Merge and split the 8 shards in `data_batches/` into `dataset_exam_generator.jsonl` (96 records), `dataset_train.jsonl` (86 records), and `dataset_val.jsonl` (10 records):

```bash
python prepare_dataset.py
```

Validate all generated JSONL files (checks schema, zero answer leaks on student sheets, unrequested Bloom tags, forbidden option phrases, option counts per level, and duplicates):

```bash
python validate_dataset.py dataset_exam_generator.jsonl dataset_train.jsonl dataset_val.jsonl
```

---

## Step 2: Google Colab Distillation (T4 GPU)

Open `FineTune_Qwen_Exam_Generator.ipynb` in Google Colab with a **T4 GPU** runtime:

1. Upload `dataset_train.jsonl` and `dataset_val.jsonl` (or mount Google Drive).
2. The notebook loads `unsloth/Qwen3.5-2B` with **16-bit LoRA** (`load_in_4bit = False`) and trains using `train_on_responses_only` so loss is computed only on assistant exam outputs.
3. Export the fine-tuned model to **GGUF (`q4_k_m`)** for local deployment (or save the PEFT LoRA adapter).

---

## Step 3: Local Deployment & Testing

### Option A: Ollama (Recommended)

Generate a `Modelfile` from `Modelfile.template` pointing to your exported `.gguf` file, then create the Ollama model:

```bash
python local_runner.py --write-modelfile ./qwen3.5_2b_exam_gguf/Qwen3.5-2B.Q4_K_M.gguf
ollama create qwen3.5-exam -f Modelfile
```

Verify readiness, generate an exam, and evaluate against the validation split:

```bash
python local_runner.py --backend ollama --model qwen3.5-exam --check

python local_runner.py --backend ollama --model qwen3.5-exam \
  --topic "Graf dan Pohon" --level Kuliah --bloom C3-C4 --formats Mixed --num 4 --lang id \
  --save-dir outputs --validate

python local_runner.py --backend ollama --model qwen3.5-exam --eval-val dataset_val.jsonl
```

### Option B: OpenAI-Compatible Local Server (`llama-server` / LM Studio)

Start your local server (e.g., `llama-server -m Qwen3.5-2B.Q4_K_M.gguf --port 8080`) and run:

```bash
python local_runner.py --backend openai --base-url http://localhost:8080/v1 --check

python local_runner.py --backend openai --base-url http://localhost:8080/v1 \
  --topic "Dynamic Programming" --level University --bloom C4-C6 --formats "Case Study" --num 2 --lang en \
  --validate
```

### Option C: Direct HuggingFace / PEFT

Requires `transformers>=5.2.0` (from `requirements.txt`) for Qwen3.5 weights:

```bash
python local_runner.py --backend hf --model-path unsloth/Qwen3.5-2B --adapter-path ./lora_adapter \
  --topic "Termodinamika" --level SMA --bloom C3-C4 --formats MCQ --num 5 --lang id --validate
```

---

## Running Unit Tests

Run the full unit test suite (including schema, validator, dataset invariants, mocked runner backends, notebook syntax/config, and workspace hygiene):

```bash
python -m unittest discover -s tests -v
```
