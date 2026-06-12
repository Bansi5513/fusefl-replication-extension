# Replication and Extension of FuseFL

**Replication and Extension of FuseFL: Demystifying Faulty Code with LLM Step-by-Step Reasoning for Explainable Fault Localization**

Bansi Patel, Nishi Shah

---

## What This Repository Contains

This repository contains the implementation, evaluation scripts, and experimental artifacts for the replication and extension of the [FuseFL framework](https://arxiv.org/abs/2403.10507) (Widyasari et al., 2024).

We reproduce FuseFL results using GPT-3.5 and extend to multiple LLMs via Groq API:
- Llama 3.3 70B
- Llama 3.1 8B
- Qwen3 32B
- GPT-OSS 20B

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
│   ├── results_summary.txt              # Replication results (GPT-3.5 + baselines)
│   ├── multi_model_comparison.txt       # Full multi-model Top-K results
│   ├── results_extended_summary.txt     # Extended results summary
│   ├── topk_comparison.png              # Top-K bar chart (replication)
│   ├── improvement_comparison.png       # Improvement chart (replication)
│   ├── topk_extended.png                # Top-K bar chart (all models)
│   ├── improvement_extended.png
│   ├── ensemble_voting.png
│   ├── error_type_analysis.png
│   └── explanation_complexity.png
│
└── dataset/
    └── README.md   
```

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:
```bash
pip install scipy numpy pandas matplotlib groq
```

**For the Groq extension**, you need a free Groq API key from [console.groq.com](https://console.groq.com). Set it as an environment variable.

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

## Original FuseFL Paper

This work replicates and extends:

> Widyasari, R., Ang, J. W., Nguyen, T. G., Sharma, N., & Lo, D. (2024). Demystifying faulty code with LLM: Step-by-step reasoning for explainable fault localization. *arXiv:2403.10507*

The pre-stored FuseFL prompt files and GPT-3.5 result files used as baselines are from the original FuseFL replication package.

---
