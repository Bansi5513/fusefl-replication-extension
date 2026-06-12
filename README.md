# Replication and Extension of FuseFL

**Replication and Extension of FuseFL: Demystifying Faulty Code with LLM Step-by-Step Reasoning for Explainable Fault Localization**

Bansi Patel, Nishi Shah

---

## What This Repository Contains

This repository contains all scripts, results, and sample data for our independent replication and extension of the [FuseFL framework](https://arxiv.org/abs/2403.10507) (Widyasari et al., 2024).

We replicate FuseFL's fault localization evaluation on the Refactory dataset using GPT-3.5, then extend it to four open-source models via the Groq inference API: Llama 3.3 70B, Llama 3.1 8B, Qwen3 32B, and GPT-OSS 20B. We also introduce an iterative refinement framework that improves the quality of generated fault explanations through self-critique.

---

## Key Results

| Model | Top-1 | Top-1% |
|---|---|---|
| Ochiai SBFL (baseline) | 157 | 48.5% |
| LLM Baseline (Wu et al.) | 150 | 46.3% |
| FuseFL GPT-3.5 (replicated) | 197 | 60.8% |
| **Llama 3.3 70B (Groq)** | **218** | **67.3%** |
| Llama 3.1 8B (Groq) | 214 | 66.0% |
| **Qwen3 32B (Groq)** | **218** | **67.3%** |
| GPT-OSS 20B (Groq) | 186 | 57.4% |

Three of four Groq-hosted open-source models exceed FuseFL (GPT-3.5) at Top-1. Llama 3.3 70B's advantage is statistically significant (p = 0.046).

---

## Repository Structure

```
fusefl-replication-extension/
│
├── README.md
│
├── scripts/
│   ├── evaluate.py              # Replication pipeline: computes Top-K scores and statistics
│   ├── multi_model_runner.py    # Runs all four Groq models using the FuseFL prompt
│   ├── compare_models.py        # Generates comparison tables and charts across all models
│   └── evaluate_extended.py    # Extended evaluation with additional metrics
│
├── results/
│   ├── results_summary.txt                    # Replication results (GPT-3.5 + baselines)
│   ├── multi_model_comparison.txt             # Full multi-model Top-K results
│   ├── results_extended_summary.txt           # Extended results summary
│   ├── replication/
│   │   ├── topk_comparison.png
│   │   └── improvement_comparison.png
│   └── extension/
│       ├── topk_comparison.png
│       ├── improvement_comparison.png
│       ├── ensemble_voting.png
│       ├── error_type_analysis.png
│       └── explanation_complexity.png
│
└── dataset/
    └── sample/
        └── question_1/          # Sample of 33 faulty Python submissions from Refactory
```

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:
```bash
pip install scipy numpy pandas matplotlib groq
```

**For the Groq extension**, you need a free Groq API key from [console.groq.com](https://console.groq.com). Set it as an environment variable — never paste it directly into the script:

On Mac/Linux:
```bash
export GROQ_API_KEY="your_key_here"
```

On Windows (Command Prompt):
```
set GROQ_API_KEY=your_key_here
```

---

## How to Run

**1. Replication (GPT-3.5 results):**
```bash
python scripts/evaluate.py
```
This loads pre-stored FuseFL result files and computes Top-K scores, Wilcoxon p-values, and Cohen's d effect sizes. No API key needed.

**2. Groq extension (run all four open-source models):**
```bash
python scripts/multi_model_runner.py
```
Requires `GROQ_API_KEY` to be set. Runs Llama 3.3 70B, Llama 3.1 8B, Qwen3 32B, and GPT-OSS 20B against the FuseFL prompt on all 324 Refactory files.

**3. Generate comparison charts:**
```bash
python scripts/compare_models.py
```
Produces the Top-K bar chart, improvement chart, and Cohen's d heatmap.

---

## Dataset

The sample in `dataset/sample/question_1/` contains 33 faulty Python submissions from the [Refactory dataset](https://github.com/githubhuyang/refactory) (Hu et al., 2019). The full dataset (1,783 files) is available at that link.

---

## Original FuseFL Paper

This work replicates and extends:

> Widyasari, R., Ang, J. W., Nguyen, T. G., Sharma, N., & Lo, D. (2024). Demystifying faulty code with LLM: Step-by-step reasoning for explainable fault localization. *arXiv:2403.10507*

The pre-stored FuseFL prompt files and GPT-3.5 result files used as baselines are from the original FuseFL replication package.

---
