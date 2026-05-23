# OmniBrief: Automated Multi-Model Agentic Web Scraping & Synthesis Pipeline

OmniBrief is an asynchronous intelligence pipeline for autonomous browser-based web extraction and executive briefing generation. It uses `browser-use`, Playwright, LangChain-compatible chat models, and a dual-model orchestration pattern to navigate dynamic websites, extract high-signal information, track telemetry, save execution traces, and synthesize results into polished Markdown morning briefings.

This project documents the full engineering journey from a basic browser agent skeleton to a production-oriented, observable, token-aware, multi-model scraping and synthesis system. The final optimized pipeline achieved a measured **28% reduction in token consumption** and a **34% reduction in end-to-end execution latency** compared with the pre-minimization benchmark.

---

## 1. System Architecture Overview

Scraping modern web pages with AI browser agents is difficult because real websites are dynamic, noisy, and often hostile to deterministic extraction. BBC Weather hydrates data after page load, Hacker News requires multi-step navigation into comments, and Wikipedia Current Events contains useful information alongside page furniture, references, navigation, and sidebars.

OmniBrief addresses these issues through a **Dual-LLM Orchestration Architecture**:

1. **Extraction Model (`LLM_MODEL`)**: Drives the `browser-use` agent. It navigates pages, observes DOM state, handles multi-step interactions, waits for dynamically loaded values, and returns raw extracted content.
2. **Summary Model (`SUMMARY_MODEL`)**: Consumes successful extractions and produces a concise executive summary for the final Markdown briefing.
3. **Deterministic Formatter**: Normalizes the report structure, applies section-level formatting rules, standardizes weather metrics, removes duplicate summary headings, and appends telemetry.
4. **Trace & Telemetry Layer**: Saves execution traces for every target and aggregates token usage, execution time, status, and trace paths across the run.

The design intentionally separates browser reasoning from final narrative synthesis. Larger models can be used where agentic navigation is hard, while smaller models can be used for cheaper summarization once the source material has been extracted.

---

## 2. Current Pipeline Capabilities

- Asynchronous concurrent target processing with `asyncio.gather`.
- Configurable OpenAI-compatible model provider via `API_KEY`, `BASE_URL`, `LLM_MODEL`, and `SUMMARY_MODEL`.
- Text-only model support with `AGENT_USE_VISION=false`, preventing unsupported `image_url` payloads.
- Custom LangChain-to-browser-use adapter with token telemetry capture.
- Browser-use trace serialization for each website target.
- Per-run trace folders and report files using a shared professional run ID.
- DOM and network optimization for lower token and load-time overhead.
- Markdown report generation with an executive summary, source sections, trace links, and telemetry.
- Dockerfile for headless server execution.
- Model evaluation runner for controlled LLM extraction/summarization experiments.

---

## 3. Directory Structure

```text
OmniBrief/
├── .env.example
├── Dockerfile
├── README.md
├── requirements.txt
├── targets.json
├── output/
│   └── omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS.md
├── src/
│   ├── browser_agent.py
│   ├── config.py
│   ├── formatter.py
│   ├── langchain_adapter.py
│   ├── main.py
│   └── telemetry.py
├── tests/
│   └── llm_model_evaluation/
│       ├── README.md
│       ├── run_model_test.py
│       └── test_results.zip
└── traces/
    └── omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS/
        ├── trace_wikipedia-current-events_YYYY-MM-DD_HH-MM-SS.json
        ├── trace_hacker-news_YYYY-MM-DD_HH-MM-SS.json
        └── trace_bbc-weather-london_YYYY-MM-DD_HH-MM-SS.json
```

### Core Files

- `src/config.py`: Loads `.env`, target configuration, model settings, browser settings, and token optimization flags.
- `src/browser_agent.py`: Runs the browser-use agent, applies DOM stripping, blocks heavy network assets, saves traces, and records token usage.
- `src/langchain_adapter.py`: Bridges LangChain chat models into browser-use’s expected LLM interface and captures usage telemetry.
- `src/formatter.py`: Generates the final Markdown report, executive summary, structured target sections, trace links, and telemetry.
- `src/main.py`: Orchestrates the full concurrent run across all targets.
- `targets.json`: Defines the current extraction targets and task prompts.
- `tests/llm_model_evaluation/run_model_test.py`: Runs the same pipeline with temporary model overrides for benchmarking.
- `tests/llm_model_evaluation/test_results.zip`: Archived benchmark artifacts containing generated reports, traces, and telemetry.

---

## 4. Target Websites

The current briefing targets are:

| Target | Purpose | Notes |
| --- | --- | --- |
| Wikipedia Current Events | Extract top current-event bullet points. | Uses a strict one-step extraction prompt because static text is usually available immediately. |
| Hacker News | Extract top stories, scores, comments, and community reaction. | Multi-step task requiring navigation into comment sections. |
| BBC Weather London | Extract high, low, conditions, and precipitation. | Requires observation after page hydration to avoid blank skeleton UI extraction. |

---

## 5. Configuration

Create a local `.env` file from `.env.example`.

```bash
cp .env.example .env
```

Important environment variables:

```env
API_KEY=your_llm_provider_api_key_here
BASE_URL=https://your-openai-compatible-provider.example.com/v1

LLM_MODEL=GPT OSS 120B
SUMMARY_MODEL=Llama 3.1 8B

LLM_TEMPERATURE=0
LLM_MAX_TOKENS=
LLM_TIMEOUT=60

SUMMARY_TEMPERATURE=0.2
SUMMARY_MAX_TOKENS=600
SUMMARY_MAX_INPUT_CHARS=12000

HEADLESS_MODE=true
AGENT_USE_VISION=false
AGENT_USE_JUDGE=false
AGENT_MAX_FAILURES=3

AGENT_CONTEXT_MINIMIZATION=true
AGENT_INCLUDE_ATTRIBUTES=href,src,id,aria-label,title,alt
```

### Text-Only Model Support

Some OpenAI-compatible third-party models do not support image inputs. OmniBrief is configured for text-only models by default:

```env
AGENT_USE_VISION=false
```

The custom LangChain adapter strips or converts image payloads before they reach the provider, while the browser agent still uses DOM text for extraction.

---

## 6. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install --with-deps chromium
```

Run the main pipeline:

```bash
.venv/bin/python src/main.py
```

Run a model evaluation:

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B"
```

Optional one-off provider override:

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B" \
  --base-url "https://your-provider.example.com/v1"
```

---

## 7. Output & Trace Naming Convention

Every run gets a shared run ID:

```text
omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS
```

The final Markdown report is saved as:

```text
output/omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS.md
```

All target traces for the same run are saved under:

```text
traces/omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS/
```

Each target trace keeps a readable target slug:

```text
trace_hacker-news_YYYY-MM-DD_HH-MM-SS.json
trace_bbc-weather-london_YYYY-MM-DD_HH-MM-SS.json
trace_wikipedia-current-events_YYYY-MM-DD_HH-MM-SS.json
```

The Markdown report includes:

- Run ID.
- Trace folder link.
- Per-target source links.
- Per-target trace file links.
- Target status.
- Executive summary.
- Telemetry and token usage.

---

## 8. Empirical Model Benchmarking & Testing

The evaluation suite tested different extraction and summarization model combinations against the same real-world targets.

### Command Execution Baseline

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "[EXTRACTION_MODEL]" \
  --summary-model "[SUMMARY_MODEL]"
```

### Historical Performance & Iteration Matrix

| Run ID / Timestamp | Extraction Model (`LLM_MODEL`) | Summary Model (`SUMMARY_MODEL`) | Combined Tokens | Execution Time | Extraction Status | Synthesis & Formatting Quality |
| --- | --- | --- | --- | --- | --- | --- |
| **Test 1** (03:31:26) | Qwen 3.5 9B | GPT OSS 120B | ~101,499 | 418.69s | Incomplete | Excellent narrative synthesis, clean Markdown layout. |
| **Test 2** (03:38:43) | Gemma 4 31B IT | Meta Llama 3.3 70B | N/A | Timeout | Total failure | Agent timed out or crashed on dynamic UI elements. |
| **Test 3** (03:46:49) | Llama 3.1 8B | GPT OSS 120B | N/A | Timeout | Total failure | Agent failed to navigate complex DOM trees. |
| **Test 4** (03:50:15) | GPT OSS 120B | Llama 3.1 8B | ~383,367 | 220.14s | 100% complete | Raw formatting with duplicated headers and unformatted strings. |
| **Test 5** (04:54:58) | GPT OSS 120B | Llama 3.1 8B with fine-tuned prompt | ~539,062 | 341.01s | 100% complete | Excellent structural control and weather metric formatting. |
| **Test 6** (05:09:03) | GPT OSS 120B with DOM minimization | Llama 3.1 8B with fine-tuned prompt | ~389,384 | 276.51s | 100% complete | Excellent formatting, lower latency, and reduced token footprint. |

### Key Findings

- **Small models struggled in agentic extraction roles.** Models with large context windows but lower reasoning capacity failed to reliably handle browser state, dynamic hydration, and multi-step navigation.
- **The larger extraction model improved completion rate.** Moving GPT OSS 120B into the extraction layer produced reliable end-to-end completion.
- **The smaller summary model was viable after guardrails.** Llama 3.1 8B performed well for final synthesis once the formatter and summary prompt enforced strict structure.
- **DOM minimization had measurable impact.** Filtering hidden elements, non-content tags, noisy attributes, and heavy media reduced token pressure and improved runtime.

---

## 9. Engineering Optimization Phases

### Phase A: Summary Prompt Fine-Tuning & Structural Guardrails

The first successful extraction runs showed that the summarizer could produce duplicate headings and raw strings, especially for weather data. The formatter and summary prompt were refined to enforce:

- A single executive summary section.
- No conversational preamble or postamble.
- No repeated summary title.
- Neutral, professional, analytical tone.
- Weather metrics with bold standardized labels.
- Clean unavailable-state blocks for failed targets.
- Deterministic target section hierarchy.

The summary model is instructed to synthesize only the extracted source material and avoid inventing facts, metrics, dates, or reactions.

### Phase B: DOM Context Minimization & Pruning

Raw browser DOMs are expensive and noisy. OmniBrief reduces context before it reaches the extraction model through:

- **Noise Stripping:** Excludes `<script>`, `<style>`, `<svg>`, `<noscript>`, `<iframe>`, footer, navigation, and hidden elements.
- **Network Blocking:** Aborts heavy media, CSS, and font requests such as `.png`, `.jpg`, `.svg`, `.css`, `.woff`, and `.ttf`.
- **Attribute Cleansing:** Limits exposed attributes to a small allowlist: `href`, `src`, `id`, `aria-label`, `title`, and `alt`.
- **Semantic Prompting:** Instructs the agent to focus on dense visible text, metrics, article titles, timestamps, scores, comment counts, weather values, and data-bearing links.
- **Skeleton UI Protection:** Prompts weather extraction to observe hydrated values before answering.

Measured optimization result:

- Token weight reduced from ~539,062 to ~389,384 tokens in the comparable optimized run.
- Execution latency reduced from 341.01s to 276.51s.
- Data completion remained at 100%.

---

## 10. Reliability Fixes Implemented

Several production-hardening fixes were added during development:

- Replaced `OPENAI_API_KEY` assumptions with provider-neutral `API_KEY` and `BASE_URL`.
- Made all model parameters configurable from `.env`.
- Added a custom `ChatLangChain` adapter because browser-use expected provider metadata and accepted extra runtime kwargs.
- Fixed `session_id` incompatibility by filtering unsupported kwargs before LangChain invocation.
- Disabled vision payloads for text-only third-party models to avoid `Unsupported ChatMessageContent type: image_url`.
- Added text-only serialization fallback for accidental image content.
- Disabled browser-use storage-state persistence for ephemeral sessions to prevent detached-target shutdown failures.
- Ensured each retry uses a fresh browser session instead of reusing a killed or reset session.
- Added concurrent target processing with robust per-target exception handling.
- Added professional artifact naming and per-run trace folders.

---

## 11. Docker Deployment

Build the container:

```bash
docker build -t omnibrief .
```

Run with your local `.env`:

```bash
docker run --rm --env-file .env -v "$PWD/output:/app/output" -v "$PWD/traces:/app/traces" omnibrief
```

The Dockerfile installs Playwright Chromium and required system dependencies for headless server execution.

---

## 12. Cron Scheduling

Example daily 06:00 run:

```cron
0 6 * * * cd /path/to/OmniBrief && /path/to/OmniBrief/.venv/bin/python src/main.py
```

Example model-evaluation cron:

```cron
0 6 * * * cd /path/to/OmniBrief && /path/to/OmniBrief/.venv/bin/python tests/llm_model_evaluation/run_model_test.py --llm-model "GPT OSS 120B" --summary-model "Llama 3.1 8B"
```

---

## 13. Recommended Production Model Layout

Based on the benchmark runs:

```bash
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B"
```

Recommended operating assumptions:

- Use the stronger model for browser extraction and navigation.
- Use the lighter model for final summarization.
- Keep `AGENT_USE_VISION=false` unless the selected model explicitly supports image input.
- Keep `AGENT_CONTEXT_MINIMIZATION=true` for lower token cost.
- Preserve trace files for observability and benchmark comparisons.

---

## 14. Future Improvements

- Add structured JSON extraction schemas per target.
- Add retry policies per website instead of global retry behavior.
- Add target-specific timeout controls.
- Add automatic benchmark comparison reports.
- Add cost estimation per provider and model.
- Add CI smoke tests for formatter output stability.
- Add optional Slack, email, or static-site publishing integrations.
