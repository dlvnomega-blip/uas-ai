"""Multi-backend local inference and validation runner for Qwen3.5-2B Exam Distillation.

Backends:
  - ollama (default): Ollama /api/chat (http://localhost:11434)
  - openai: OpenAI-compatible local server (llama-server, LM Studio, vLLM)
  - hf: Direct HuggingFace transformers (+ optional PEFT LoRA adapter)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from schema import (
    BLOOM_PROFILES,
    CANONICAL_SYSTEM_PROMPT,
    EDUCATION_LEVELS,
    QUESTION_FORMATS,
    build_user_prompt,
    parse_exam_output,
    strip_thinking,
)
from validate_dataset import validate_record

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

EMBEDDED_MODELFILE_TEMPLATE = """FROM {{GGUF_PATH}}

TEMPLATE \"\"\"{{- if .Messages }}{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
<think>

</think>

{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}{{- end }}{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
<think>

</think>

{{ end }}{{- end }}{{- else }}{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
<think>

</think>

{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}\"\"\"

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER num_ctx 4096
PARAMETER temperature 0.6
PARAMETER top_p 0.85
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
"""


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Perform a GET (payload=None) or POST JSON request using stdlib urllib."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if payload is not None else "GET"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_openai_v1(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _check_ollama(base_url: str, model: str) -> int:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        data = _http_json(url, timeout=5.0)
    except Exception as exc:
        print(f"[ollama] NOT READY at {base_url}: {exc}", file=sys.stderr)
        return 1
    models = [m.get("name", "") for m in data.get("models", [])]
    found = any(m == model or m.startswith(f"{model}:") for m in models)
    print(f"[ollama] READY at {base_url} | models={len(models)} | target '{model}' present={found}")
    return 0


def _generate_ollama(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool = True,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 8192,
            "num_predict": max_tokens,
        },
    }
    if not stream:
        data = _http_json(url, payload=payload, timeout=600.0)
        return strip_thinking(data.get("message", {}).get("content", ""))

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            obj = json.loads(line)
            piece = obj.get("message", {}).get("content", "")
            if piece:
                chunks.append(piece)
                print(piece, end="", flush=True)
            if obj.get("done"):
                break
    if chunks:
        print()
    return strip_thinking("".join(chunks))


def _check_openai(base_url: str, model: str) -> int:
    v1 = _normalize_openai_v1(base_url)
    try:
        data = _http_json(f"{v1}/models", timeout=5.0)
    except Exception as exc:
        print(f"[openai] NOT READY at {v1}: {exc}", file=sys.stderr)
        return 1
    models = [m.get("id", "") for m in data.get("data", [])]
    print(f"[openai] READY at {v1} | models={models} | target='{model}'")
    return 0


def _generate_openai(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    v1 = _normalize_openai_v1(base_url)
    url = f"{v1}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        data = _http_json(url, payload=payload, timeout=600.0)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 422):
            payload.pop("chat_template_kwargs", None)
            data = _http_json(url, payload=payload, timeout=600.0)
        else:
            raise
    content = data["choices"][0]["message"]["content"]
    return strip_thinking(content)


def _check_hf(adapter_path: str | None) -> int:
    missing: list[str] = []
    versions: dict[str, str] = {}
    for pkg in ("torch", "transformers"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            missing.append(pkg)
    try:
        peft_mod = __import__("peft")
        versions["peft"] = getattr(peft_mod, "__version__", "unknown")
    except ImportError:
        versions["peft"] = "not installed"
        if adapter_path:
            missing.append("peft")

    if missing:
        print(f"[hf] NOT READY: missing packages: {', '.join(missing)}", file=sys.stderr)
        return 1

    import torch

    cuda_ok = torch.cuda.is_available()
    vram_info = ""
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        vram_info = f" | GPU={props.name} ({vram_gb:.1f} GB VRAM)"
    print(
        f"[hf] READY | torch={versions['torch']} transformers={versions['transformers']} "
        f"peft={versions['peft']} | cuda={cuda_ok}{vram_info}"
    )
    ver_parts = tuple(int(x) for x in re.findall(r"\d+", versions["transformers"])[:3])
    if ver_parts and ver_parts < (5, 2, 0):
        print(
            f"[hf] NOTE: Qwen3.5 weights require transformers>=5.2.0 (found {versions['transformers']}).",
            file=sys.stderr,
        )
    return 0


class _HFRunner:
    def __init__(self, model_path: str, adapter_path: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map: str | None = "auto"
        else:
            dtype = torch.float32
            device_map = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.torch = torch

    def generate(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][input_len:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return strip_thinking(decoded)


def _read_reference_file(ref_path: str) -> str:
    path = Path(ref_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required to read .pdf reference files (pip install pypdf)"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    return path.read_text(encoding="utf-8-sig").strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:60] or "exam"


def _write_modelfile(gguf_path: str, out_path: str = "Modelfile") -> None:
    tpl_path = Path("Modelfile.template")
    if tpl_path.exists():
        content = tpl_path.read_text(encoding="utf-8")
    else:
        content = EMBEDDED_MODELFILE_TEMPLATE.replace("{SYSTEM_PROMPT}", CANONICAL_SYSTEM_PROMPT)
    content = content.replace("{{GGUF_PATH}}", gguf_path).replace("{gguf_path}", gguf_path)
    Path(out_path).write_text(content, encoding="utf-8")
    print(f"Wrote {out_path} pointing to {gguf_path}")


def _run_generation(
    backend: str,
    model: str,
    base_url: str,
    hf_runner: _HFRunner | None,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> str:
    if backend == "ollama":
        return _generate_ollama(
            base_url, model, messages, temperature, max_tokens, stream=stream
        )
    if backend == "openai":
        return _generate_openai(base_url, model, messages, temperature, max_tokens)
    if backend == "hf":
        assert hf_runner is not None
        return hf_runner.generate(messages, temperature, max_tokens)
    raise ValueError(f"Unsupported backend: {backend}")


def _eval_validation_set(
    jsonl_path: str,
    limit: int | None,
    backend: str,
    model: str,
    base_url: str,
    hf_runner: _HFRunner | None,
    temperature: float,
    max_tokens: int,
) -> int:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Validation file not found: {jsonl_path}", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break

    if not records:
        print("No records found in validation file.", file=sys.stderr)
        return 1

    passed = 0
    timings: list[float] = []
    for idx, rec in enumerate(records):
        msgs = rec.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), CANONICAL_SYSTEM_PROMPT)
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        prompt_msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        t0 = time.perf_counter()
        output = _run_generation(
            backend, model, base_url, hf_runner, prompt_msgs, temperature, max_tokens, stream=False
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        parsed = parse_exam_output(output)
        eval_record = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": parsed["raw_clean"]},
            ]
        }
        errors, warnings, _ = validate_record(eval_record, idx=idx)
        ok = len(errors) == 0 and bool(parsed["student_sheet"]) and bool(parsed["answer_key"])
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{idx + 1}/{len(records)}] {status} ({elapsed:.2f}s) errors={len(errors)} warnings={len(warnings)}")
        if errors:
            for err in errors[:3]:
                print(f"    ERR: {err}")

    avg_t = sum(timings) / len(timings)
    rate = 100.0 * passed / len(records)
    print(f"\nSummary: {passed}/{len(records)} passed ({rate:.1f}%) | avg={avg_t:.2f}s | total={sum(timings):.2f}s")
    return 0 if passed == len(records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local inference and validation runner for distilled Qwen3.5-2B exam models."
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "openai", "hf"),
        default="ollama",
        help="Inference backend (default: ollama).",
    )
    parser.add_argument("--model", default=None, help="Model name for ollama/openai.")
    parser.add_argument("--model-path", default=None, help="HF model directory or repo ID.")
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT LoRA adapter path for hf.")
    parser.add_argument("--base-url", default=None, help="Server URL for ollama or openai.")
    parser.add_argument("--check", action="store_true", help="Probe backend readiness and exit.")

    parser.add_argument("--topic", default=None, help="Exam topic.")
    parser.add_argument("--level", default="Kuliah", choices=EDUCATION_LEVELS, help="Education level.")
    parser.add_argument("--bloom", default="C1-C6", choices=BLOOM_PROFILES, help="Bloom profile.")
    parser.add_argument("--formats", default="Mixed", choices=QUESTION_FORMATS, help="Question formats.")
    parser.add_argument("--num", type=int, default=3, help="Number of questions (default: 3).")
    parser.add_argument("--lang", default="id", choices=("id", "en"), help="Language (id or en).")
    parser.add_argument("--ref-file", default=None, help="Optional reference file (.txt, .md, .pdf).")
    parser.add_argument("--instructions", default=None, help="Optional custom instructions.")
    parser.add_argument("--show-bloom", action="store_true", help="Include Bloom labels on student sheet.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6).")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generated tokens (default: 2048).")
    parser.add_argument("--save-dir", default=None, help="Directory to save <slug>_student.md and <slug>_key.md.")
    parser.add_argument("--validate", action="store_true", help="Run validate_record on generated output.")

    parser.add_argument("--eval-val", default=None, help="Path to validation JSONL for batch evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Max records to evaluate in --eval-val.")
    parser.add_argument("--write-modelfile", default=None, help="Write Ollama Modelfile for the given GGUF path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.write_modelfile:
        _write_modelfile(args.write_modelfile)
        return 0

    base_url = args.base_url or (
        "http://localhost:11434" if args.backend == "ollama" else "http://localhost:8080/v1"
    )
    model = args.model or ("qwen3.5-exam" if args.backend == "ollama" else "qwen3.5-2b")

    if args.check:
        if args.backend == "ollama":
            return _check_ollama(base_url, model)
        if args.backend == "openai":
            return _check_openai(base_url, model)
        return _check_hf(args.adapter_path)

    hf_runner: _HFRunner | None = None
    if args.backend == "hf":
        if not args.model_path:
            parser.error("--model-path is required when --backend hf is used.")
        hf_runner = _HFRunner(args.model_path, args.adapter_path)

    if args.eval_val:
        return _eval_validation_set(
            args.eval_val,
            args.limit,
            args.backend,
            model,
            base_url,
            hf_runner,
            args.temperature,
            args.max_tokens,
        )

    if not args.topic:
        parser.error("--topic is required for generation (or use --check / --eval-val / --write-modelfile).")

    ref_text = _read_reference_file(args.ref_file) if args.ref_file else None
    user_prompt = build_user_prompt(
        topic=args.topic,
        education_level=args.level,
        bloom_profile=args.bloom,
        question_formats=args.formats,
        num_questions=args.num,
        language=args.lang,
        reference_material=ref_text,
        custom_instructions=args.instructions,
        show_bloom_labels=args.show_bloom,
    )
    messages = [
        {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    raw_output = _run_generation(
        args.backend,
        model,
        base_url,
        hf_runner,
        messages,
        args.temperature,
        args.max_tokens,
        stream=(args.backend == "ollama"),
    )
    elapsed = time.perf_counter() - t0
    if args.backend != "ollama":
        print(raw_output)
    print(f"\n[Generated in {elapsed:.2f}s via {args.backend}]", file=sys.stderr)

    parsed = parse_exam_output(raw_output)
    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(args.topic)
        stu_path = out_dir / f"{slug}_student.md"
        key_path = out_dir / f"{slug}_key.md"
        stu_path.write_text(parsed["student_sheet"] or parsed["raw_clean"], encoding="utf-8")
        key_path.write_text(parsed["answer_key"], encoding="utf-8")
        print(f"Saved: {stu_path} and {key_path}", file=sys.stderr)

    if args.validate:
        record = {
            "messages": [
                {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": parsed["raw_clean"]},
            ]
        }
        errors, warnings, _ = validate_record(record, idx=0)
        print(f"Validation: errors={len(errors)}, warnings={len(warnings)}", file=sys.stderr)
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        for warn in warnings:
            print(f"  WARN: {warn}", file=sys.stderr)
        return 0 if not errors else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
