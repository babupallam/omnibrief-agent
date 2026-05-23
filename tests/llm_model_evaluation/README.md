# Omnibrief: LLM Model Extraction & Synthesis Evaluation

This document outlines the testing, analysis, and conclusions drawn from evaluating different combinations of Large Language Models (LLMs) for an automated web-scraping and summarization pipeline (Omnibrief).

> **Note on Raw Data:** The complete, raw test results, trace files, and final markdown outputs for all tests detailed below can be found in `tests/llm_model_evaluation/test_results.zip`.

## 1. System Architecture Overview

The Omnibrief system uses a two-model architecture to optimize for both cost and capability:

1. **Extraction Model (`LLM_MODEL`)**: Responsible for acting as an agent to navigate dynamic DOMs (Document Object Models), extract specific data (like weather or Hacker News top posts), and handle massive HTML context windows.
2. **Summary Model (`SUMMARY_MODEL`)**: Takes the raw JSON/Markdown data extracted by the first model and synthesizes it into a cohesive, highly readable morning briefing.

## 2. Test Methodology

Four tests were run consecutively to evaluate different combinations of models based on their parameter size, context windows, and theoretical capabilities. The goal was to find the optimal balance of speed, reliability, formatting quality, and cost.

### Test Cases

**Test 1 (03:31:26): Qwen / GPT OSS**

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "Qwen 3.5 9B" \
  --summary-model "GPT OSS 120B"

```

**Test 2 (03:38:43): Gemma / Meta Llama**

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "Gemma 4 31B IT" \
  --summary-model "Meta Llama 3.3 70B Instruct"

```

**Test 3 (03:46:49): Llama 8B / GPT OSS**

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "Llama 3.1 8B" \
  --summary-model "GPT OSS 120B"

```

**Test 4 (03:50:15): GPT OSS / Llama 8B**

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B"

```

## 3. Analysis & Results

The initial hypothesis was that a smaller model with a massive context window (like Qwen 3.5 9B with a 262k context or Gemma 4 31B with a 256k context) would be ideal for the heavy lifting of raw HTML extraction, while a larger model would handle the final writing.

**The practical results entirely disproved this hypothesis.** Agentic DOM navigation requires significant reasoning capabilities that small models currently lack.

### Breakdown of Results

| Test # | Extraction Model | Summary Model | Time (s) | Result Status | Notes |
| --- | --- | --- | --- | --- | --- |
| **Test 1** | Qwen 3.5 9B | GPT OSS 120B | 418.69 | ⚠️ Incomplete | Failed midway through Hacker News extraction. Writing quality of completed sections was excellent. |
| **Test 2** | Gemma 4 31B IT | Meta Llama 3.3 70B | N/A | ❌ Total Failure | Agent crashed/timed out during extraction. |
| **Test 3** | Llama 3.1 8B | GPT OSS 120B | N/A | ❌ Total Failure | Agent crashed/timed out during extraction. |
| **Test 4** | GPT OSS 120B | Llama 3.1 8B | **220.14** | ✅ **100% Complete** | Successfully extracted all data. Writing formatting was slightly messy (e.g., raw strings instead of formatted weather data). |

### Key Findings

1. **Small Models Fail at Agentic Scraping:** Models in the 8B–31B range (Qwen, Llama, Gemma) struggled to reliably execute complex agentic logic (e.g., recognizing skeleton UIs, clicking into comment threads, synthesizing nested HTML). They consistently timed out or provided incomplete data.
2. **Large Models Excel at Extraction:** GPT OSS 120B, when placed in the extraction role, easily brute-forced the messy DOM and completed the task flawlessly and quickly (220 seconds).
3. **The Summary Trade-off:** Using a smaller model (Llama 3.1 8B) for the final summary step is viable, but it requires highly prescriptive system prompts to ensure clean Markdown formatting, as it tends to output raw data strings when left unguided.

## 4. Conclusion & Final Architecture

Based on the test results, the most robust and fastest combination is **Test 4**.

To implement this into production seamlessly, we will adopt the Test 4 architecture but enhance the Summary Model's system prompt to enforce strict Markdown formatting.

### Production Configuration

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B"

```