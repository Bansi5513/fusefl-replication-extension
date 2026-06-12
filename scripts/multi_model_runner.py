import os
import re
import json
import time
import random
import requests

BASE         = os.path.dirname(os.path.abspath(__file__))  # automatically uses the folder this script is in
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  

MODELS_TO_RUN = {
    "groq_llama3_70b": True,
    "groq_llama3_8b":  True,
    "groq_qwen3":      True,
    "groq_gptoss":     True,
}

MAX_RATE_LIMIT_WAIT = 60
MAX_RETRIES         = 6
SKIP_EXISTING       = True

QUESTIONS = ['question_1', 'question_2', 'question_3',
             'question_4', 'question_5']

MODEL_CONFIGS = {
    "groq_llama3_70b": {
        "model_id": "llama-3.3-70b-versatile",
        "display":  "Llama 3.3 70B (Groq)",
        "pause":    3.5,
    },
    "groq_llama3_8b": {
        "model_id": "llama-3.1-8b-instant",
        "display":  "Llama 3.1 8B (Groq)",
        "pause":    2.0,
    },
    "groq_qwen3": {
        "model_id": "qwen/qwen3-32b",
        "display":  "Qwen3 32B (Groq)",
        "pause":    2.5,
    },
    "groq_gptoss": {
        "model_id": "openai/gpt-oss-20b",
        "display":  "GPT-OSS 20B (Groq)",
        "pause":    2.0,
    },
}


class RateLimitExceeded(Exception):
    pass


def call_model_with_retry(prompt_text, config, current_pause):
    model_id = config["model_id"]
    pause    = current_pause

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model":       model_id,
                      "messages":    [{"role": "user", "content": prompt_text}],
                      "temperature": 0.0,
                      "max_tokens":  1500},
                timeout=60,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = (float(retry_after) + random.uniform(1, 3)
                        if retry_after
                        else min(120, pause * (2 ** attempt) + random.uniform(2, 6)))
                if wait > MAX_RATE_LIMIT_WAIT:
                    print(f"    [429] Wait={wait:.0f}s > limit → skipping model")
                    raise RateLimitExceeded()
                pause = max(pause, wait / 3)
                print(f"    [429] Rate-limited — waiting {wait:.0f}s "
                      f"(attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if resp.status_code == 400:
                try:
                    msg = resp.json().get("error", {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:200]
                print(f"    [400] {msg}")
                return None, pause

            if resp.status_code == 404:
                print(f"    [404] Model not found: {model_id}")
                return None, pause

            if resp.status_code >= 500:
                wait = 10 * (attempt + 1)
                print(f"    [{resp.status_code}] Server error — retrying in {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"], pause

        except RateLimitExceeded:
            raise
        except requests.exceptions.Timeout:
            time.sleep(15 * (attempt + 1))
        except requests.exceptions.ConnectionError:
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"    [ERR attempt {attempt+1}] {e}")
            time.sleep(5 * (attempt + 1))

    return None, pause


def parse_response(raw_text):
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    for candidate in (
        re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL) +
        re.findall(r'(\{[^{}]*"faultyLine"[^{}]*\})', text, re.DOTALL) +
        re.findall(r'(\{[^{}]*"faultLoc"[^{}]*\})',   text, re.DOTALL)
    ):
        try:
            data = json.loads(candidate)
            if "faultLoc" in data:
                lines = [int(loc["faultyLine"]) for loc in data["faultLoc"]
                         if isinstance(loc, dict) and "faultyLine" in loc]
                if lines:
                    return lines
            if "faultyLine" in data:
                return [int(data["faultyLine"])]
        except Exception:
            pass

    m = re.findall(r'"faultyLine"\s*:\s*(\d+)', text)
    if m:
        return list(dict.fromkeys(int(x) for x in m))

    m = re.findall(r'(?:\d+[\.\)]\s+)?[Ll]ine\s*[:#]?\s*(\d+)', text)
    if m:
        return list(dict.fromkeys(int(x) for x in m))

    arr = re.findall(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', text)
    if arr:
        return [int(n) for n in re.findall(r'\d+', arr[0])]

    m = re.findall(r'(?m)^[ \t]*(?:[-*•]|\d+[.)]) *(\d+)\s*$', text)
    if m:
        return list(dict.fromkeys(int(x) for x in m))

    seen, result = set(), []
    for x in re.findall(r'\b(\d{1,3})\b', text):
        n = int(x)
        if n not in seen:
            seen.add(n); result.append(n)
        if len(result) == 3:
            break
    return result


def save_results(out_dir, file_key, lines, raw_text):
    os.makedirs(os.path.join(out_dir, "results_line"),  exist_ok=True)
    os.makedirs(os.path.join(out_dir, "raw_responses"), exist_ok=True)
    with open(os.path.join(out_dir, "results_line", f"{file_key}.txt"),
              "w", encoding="utf-8") as f:
        f.write(",".join(str(l) for l in lines))
    with open(os.path.join(out_dir, "raw_responses", f"{file_key}.txt"),
              "w", encoding="utf-8") as f:
        f.write(raw_text)


def count_saved(model_name):
    results_dir = os.path.join(BASE, "Prompts_Results", f"{model_name}_prompt")
    count = 0
    for q in QUESTIONS:
        line_dir = os.path.join(results_dir, q, "results_line")
        if not os.path.exists(line_dir):
            continue
        for fname in os.listdir(line_dir):
            if not fname.endswith(".txt"):
                continue
            try:
                with open(os.path.join(line_dir, fname), encoding="utf-8") as f:
                    if not f.read().strip().startswith("FAILED"):
                        count += 1
            except Exception:
                pass
    return count


def validate():
    ok = True
    if not os.path.exists(BASE):
        print(f"  ERROR: BASE folder not found: {BASE}"); ok = False
    if not os.path.exists(os.path.join(BASE, "Prompts", "FuseFL")):
        print(f"  ERROR: Prompts/FuseFL not found"); ok = False
    if not GROQ_API_KEY:
        print("  ERROR: GROQ_API_KEY is empty"); ok = False
    if ok:
        print("  All checks passed ✓")
    return ok


def run_model(model_name, config):
    display      = config["display"]
    prompts_dir  = os.path.join(BASE, "Prompts", "FuseFL")
    results_dir  = os.path.join(BASE, "Prompts_Results", f"{model_name}_prompt")
    pause        = config.get("pause", 2.5)
    saved_before = count_saved(model_name)

    print(f"\n{'─'*57}")
    print(f"  Running : {display}")
    print(f"  Model ID: {config['model_id']}")
    print(f"  Saved   : {saved_before}/324 files")
    print(f"{'─'*57}")

    if saved_before >= 324:
        print("  ✓ Already complete — skipping.")
        return 0, 0, "complete"

    total = success = failed = skipped = parse_warn = 0
    t_start = time.time()

    try:
        for question in QUESTIONS:
            q_in  = os.path.join(prompts_dir, question)
            q_out = os.path.join(results_dir, question)
            if not os.path.exists(q_in):
                continue

            files = sorted(f for f in os.listdir(q_in)
                           if os.path.isfile(os.path.join(q_in, f)))
            print(f"\n  {question}: {len(files)} files")

            for fname in files:
                key       = fname.replace('.txt', '')
                done_path = os.path.join(q_out, "results_line", f"{key}.txt")
                total    += 1

                if SKIP_EXISTING and os.path.exists(done_path):
                    try:
                        with open(done_path, encoding="utf-8") as fh:
                            if not fh.read().strip().startswith("FAILED"):
                                skipped += 1
                                continue
                    except Exception:
                        pass

                try:
                    with open(os.path.join(q_in, fname),
                              encoding='utf-8', errors='replace') as fh:
                        prompt_text = fh.read()
                except Exception as e:
                    print(f"    [READ ERR] {fname}: {e}")
                    failed += 1
                    continue

                raw, pause = call_model_with_retry(prompt_text, config, pause)

                if raw is None:
                    failed += 1
                    save_results(q_out, key, [], f"FAILED_AFTER_{MAX_RETRIES}_RETRIES")
                    print(f"    ✗ {key:<32} → FAILED")
                    continue

                lines = parse_response(raw)
                save_results(q_out, key, lines, raw)

                if lines:
                    success += 1
                    print(f"    ✓ {key:<32} → {lines[:3]}")
                else:
                    success    += 1
                    parse_warn += 1
                    print(f"    ? {key:<32} → parse failed")

                done = success + failed
                if done % 20 == 0 and done > 0:
                    elapsed = time.time() - t_start
                    remain  = total - skipped - done
                    print(f"    ── {done} new | "
                          f"ETA ≈ {elapsed/done*remain/60:.1f} min ──")

                time.sleep(pause)

    except RateLimitExceeded:
        saved_now = count_saved(model_name)
        print(f"\n  → Rate limit — moving to next model. "
              f"Saved: {saved_now}/324.")
        return success, failed, "rate_limited"

    saved_now = count_saved(model_name)
    elapsed   = time.time() - t_start
    print(f"\n  Done: {success} new | {skipped} skipped | "
          f"{failed} failed | {parse_warn} parse-warn | "
          f"{saved_now}/324 total | {elapsed/60:.1f} min")
    return success, failed, "done"


def main():
    print("=" * 57)
    print("  FuseFL Multi-Model Runner")
    print("=" * 57)

    print("\n[Validation]")
    if not validate():
        return

    enabled = [(n, MODEL_CONFIGS[n]) for n, on in MODELS_TO_RUN.items() if on]
    if not enabled:
        print("No models set to True in MODELS_TO_RUN.")
        return

    print(f"\n  {len(enabled)} models queued:")
    for n, c in enabled:
        saved = count_saved(n)
        done  = saved * 20 // 324
        bar   = "█" * done + "░" * (20 - done)
        print(f"    • {c['display']:<36} [{bar}] {saved}/324")

    summary = {}
    for n, c in enabled:
        ok, fail, status = run_model(n, c)
        summary[n] = (ok, fail, status)

    print("\n" + "=" * 57)
    print("  SESSION COMPLETE")
    print("=" * 57)
    all_done = True
    for n, (ok, fail, status) in summary.items():
        saved = count_saved(n)
        icon  = "✓" if saved >= 324 else "⏸"
        if saved < 324:
            all_done = False
        print(f"  {icon} {MODEL_CONFIGS[n]['display']:<40} "
              f"{saved:>3}/324  [{status}]")
    print()
    if all_done:
        print("  ✓ All complete! Run: python compare_models.py")
    else:
        print("  ⏸ Rerun to continue (resumes automatically).")
    print("=" * 57)


if __name__ == "__main__":
    main()
