# OmniBrief: AI Agent Web Extraction Case Study

OmniBrief is an engineering case study for building a reliable, observable, and token-aware AI browser agent pipeline for automated morning briefings.

The project focuses on a practical problem in modern AI automation: extracting useful information from dynamic websites is unreliable, slow, and expensive when an AI agent is given large, noisy browser pages without optimization.

OmniBrief uses `browser-use`, LangChain-compatible chat models, trace logging, telemetry, browser optimization, and deterministic Markdown formatting to extract information from selected websites and generate a structured briefing report.

For the full system architecture, implementation layers, optimization techniques, and step-by-step technical design, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Engineering Case Study

### 1.1 The Problem

Scraping dynamic websites with AI agents is difficult because modern websites are not clean text sources.

They often include:

- Deep DOM trees.
- Hidden elements.
- Navigation menus.
- Cookie banners.
- Advertisement blocks.
- Tracking scripts.
- Skeleton loading components.
- Dynamically hydrated data.
- Large amounts of styling, media, and layout markup.

A traditional scraper can fail because the required data may not be available in the initial HTML response. An AI browser agent can interact with the page, but it can become slow and token-heavy if the model receives too much noisy browser context.

The engineering problem was:

```text id="2z3qmp"
How can an AI browser agent extract reliable information from dynamic websites while reducing failures, latency, and token usage?
```

---

### 1.2 Initial Hypothesis

The first design hypothesis was cost-focused:

```text id="c3iqq7"
Use smaller and cheaper models with large context windows, such as Qwen 9B or Gemma 31B, for web extraction.
Then use a larger model only for the final executive summary.
```

This seemed reasonable because:

- Web extraction appeared to be mostly a parsing task.
- Large context windows seemed useful for messy browser DOMs.
- Smaller models appeared cheaper for repeated extraction.
- A larger model could be saved for final summary writing.

The expected architecture was:

```text id="n0qeqm"
Small model -> browser extraction
Large model -> final summary
```

---

### 1.3 Technical Discovery

Benchmarking showed that this assumption did not work reliably.

The main limitation was not only context length. The smaller models did not have enough agentic reasoning capability to handle complex browser tasks.

They struggled with:

- Navigating complex DOM trees.
- Understanding browser state.
- Waiting for dynamically loaded values.
- Avoiding skeleton UI placeholders.
- Following multi-step extraction paths.
- Separating useful content from page furniture.
- Recovering from failed observations.
- Returning concrete extracted data instead of generic page labels.

This caused repeated failures, timeouts, incomplete outputs, and unstable extraction quality.

The key technical finding was:

```text id="0n9y55"
Agentic web extraction is not just text parsing.
It is a browser-control and reasoning task.
```

The most difficult part of the pipeline was not writing the final briefing. The difficult part was reliably operating inside dynamic web pages and extracting the correct data.

---

### 1.4 Engineered Solution

The final solution changed the model-routing strategy.

Instead of using a small model for extraction and a large model for summary writing, OmniBrief uses:

```text id="2h6e74"
Powerful 120B model -> reliable browser extraction
Lean 8B model -> final synthesis with a strict engineered prompt
```

This architecture works because the most complex task is browser-based extraction. Once the extracted content is clean, reduced, and structured, the final summary becomes a much simpler task.

The final implementation combines:

- A powerful extraction model for browser navigation and dynamic page reasoning.
- A smaller summary model for final synthesis.
- A strict summary prompt to control tone, structure, and factual grounding.
- DOM context minimization to reduce noisy browser content.
- Network blocking to avoid loading unnecessary assets.
- Text-only model support for providers that do not support image inputs.
- Deterministic Markdown formatting for stable report output.
- Per-target trace files for debugging.
- Token and execution telemetry for measurement.

The optimized architecture achieved reliable extraction with **100% data completion** in the successful benchmark runs and reduced processing time significantly compared with the early inefficient or failed extraction approaches.

---

## 2. System Design

The current implementation is:

```text id="9ps8gk"
A single browser agent with layered orchestration
```

The system is not a true multi-agent architecture. The pipeline currently uses:

- One autonomous browser agent for extraction.
- Separate extraction and summary models.
- Deterministic orchestration logic.
- Concurrent target processing.
- Trace and telemetry layers.

The browser agent is responsible for:

- Navigation.
- DOM observation.
- Browser interaction.
- Extraction reasoning.
- Retry handling.

The summary stage is a separate model-routing layer, not an autonomous agent.

---

## 3. Project Outcome

OmniBrief generates a complete Markdown morning briefing from multiple web sources.

The final report includes:

- Run ID.
- Generated timestamp.
- Executive summary.
- Source-specific sections.
- Per-target extraction status.
- Source links.
- Trace file links.
- Token usage.
- Execution time telemetry.

The project demonstrates an important engineering pattern:

```text id="4vs4gi"
Use stronger models where the environment is complex.
Use smaller models where the input is already clean.
Use deterministic code where output structure must be stable.
Use telemetry where performance and cost need to be measured.
```

---

## 4. Current Pipeline Capabilities

- Asynchronous concurrent target processing using `asyncio.gather`.
- Configurable OpenAI-compatible provider support.
- Separate extraction and summary models.
- Text-only model support with `AGENT_USE_VISION=false`.
- Custom LangChain-to-browser-use adapter.
- Token telemetry capture.
- Browser-use trace serialization for every target.
- Per-run trace folders and report files.
- DOM minimization for noisy page content.
- Network blocking for heavy assets.
- Markdown briefing generation.
- Docker support for headless execution.
- Model evaluation runner for controlled experiments.

---

## 5. Repository Structure

```text id="laxd9y"
OmniBrief/
├── .env.example
├── Dockerfile
├── README.md
├── ARCHITECTURE.md
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

---

## 6. Model Evaluation and Benchmarking

The project includes controlled model evaluation experiments under:

```text id="jlwmva"
tests/llm_model_evaluation/
```

The benchmark results, prompts, execution outcomes, and model comparisons are documented in:

```text id="5e9c7y"
tests/llm_model_evaluation/README.md
```

The generated evaluation outputs and benchmark artifacts are included in:

```text id="rjlwmh"
tests/llm_model_evaluation/test_results.zip
```

These experiments were used to compare:

- Different extraction models.
- Different summary models.
- Different prompt strategies.
- Token usage.
- Execution latency.
- Completion reliability.
- Formatting quality.
- Browser extraction stability.

The evaluation process directly influenced the final architecture decisions and optimization strategy.

---

## 7. Target Websites

The current briefing targets are:

| Target | Purpose |
| --- | --- |
| Wikipedia Current Events | Extract top current-event bullet points. |
| Hacker News | Extract top stories, scores, comments, and community reaction. |
| BBC Weather London | Extract high, low, conditions, and precipitation. |

---

## 8. Configuration

Create a local `.env` file from `.env.example`.

```bash id="y8x35q"
cp .env.example .env
```

Important environment variables:

```env id="8fb0lx"
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

---

## 9. Text-Only Model Support

Some OpenAI-compatible third-party providers do not support image inputs.

OmniBrief uses text-only operation by default:

```env id="vb89o7"
AGENT_USE_VISION=false
```

Only enable vision mode if the selected provider supports image inputs:

```env id="u5tgs5"
AGENT_USE_VISION=true
```

---

## 10. Installation

Create and activate a virtual environment:

```bash id="g7o2i0"
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash id="w73o42"
pip install -r requirements.txt
```

---

## 11. Running the Main Pipeline

Run the main briefing pipeline:

```bash id="9np9vl"
.venv/bin/python src/main.py
```

The generated Markdown report will be saved under:

```text id="1o0xj7"
output/
```

Trace files will be saved under:

```text id="wq84n0"
traces/
```

---

## 12. Running a Model Evaluation

Run a model evaluation with custom extraction and summary models:

```bash id="p1ixyz"
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B"
```

Run with a temporary provider override:

```bash id="gq11kk"
.venv/bin/python tests/llm_model_evaluation/run_model_test.py \
  --llm-model "GPT OSS 120B" \
  --summary-model "Llama 3.1 8B" \
  --base-url "https://your-provider.example.com/v1"
```

---

## 13. Output and Trace Naming Convention

Every run gets a shared run ID:

```text id="ryp4iz"
omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS
```

The final Markdown report is saved as:

```text id="m5xojl"
output/omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS.md
```

All target traces for the same run are saved under:

```text id="2jmy2d"
traces/omnibrief-morning-briefing_YYYY-MM-DD_HH-MM-SS/
```

Each target trace keeps a readable target slug:

```text id="c8zn2y"
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

## 14. Docker Deployment

Build the Docker image:

```bash id="g4exxe"
docker build -t omnibrief .
```

Run with your local `.env` file:

```bash id="ywx0kx"
docker run --rm --env-file .env -v "$PWD/output:/app/output" -v "$PWD/traces:/app/traces" omnibrief
```

---

## 15. Cron Scheduling

Example daily 06:00 run:

```cron id="abk9n6"
0 6 * * * cd /path/to/OmniBrief && /path/to/OmniBrief/.venv/bin/python src/main.py
```

Example scheduled model evaluation:

```cron id="wjw9hm"
0 6 * * * cd /path/to/OmniBrief && /path/to/OmniBrief/.venv/bin/python tests/llm_model_evaluation/run_model_test.py --llm-model "GPT OSS 120B" --summary-model "Llama 3.1 8B"
```

---

## 16. Technical Value

OmniBrief is not only a web scraping project. It demonstrates:

- Model routing.
- Agent reliability engineering.
- Cost-aware LLM system design.
- Browser-based AI extraction.
- Telemetry-driven optimization.
- Token-aware orchestration.

The main engineering insight is:

```text id="h2svax"
The cheapest model is not always cheaper if it fails, loops, times out, or requires repeated retries.
```

The final design uses the right component at the right layer:

- Strong model for browser extraction.
- Lean model for final synthesis.
- DOM minimization for lower context load.
- Network blocking for faster browser operation.
- Deterministic code for report structure.
- Telemetry for performance measurement.
- Trace files for debugging and auditability.

This makes OmniBrief suitable as a portfolio case study for:

- AI automation systems.
- Agentic workflows.
- Browser-use architectures.
- Production-oriented LLM engineering.
- Model orchestration pipelines.

---

## 17. Future Directions

Potential future improvements include:

- Structured JSON extraction schemas per target.
- Target-specific retry policies.
- Target-specific timeout controls.
- Automatic benchmark comparison reports.
- Cost estimation per provider and model.
- CI smoke tests for formatter stability.
- Slack, email, or static-site publishing integrations.
- Lightweight dashboard for trace inspection.
- Per-target extraction confidence scoring.
- Automatic layout-change detection.
- Multi-agent orchestration architectures.

The current implementation uses:

```text id="j90n2u"
A single browser agent with layered orchestration
```

A future version may evolve toward:

- Planner agents.
- Verification agents.
- Critic agents.
- Recovery agents.
- Multi-agent coordination pipelines.



---

## 18. Real-World Use Cases

The architecture used in OmniBrief is not limited to morning briefings. The same design pattern can be applied to enterprise systems that require reliable extraction from dynamic web interfaces, dashboards, portals, or semi-structured online sources.

The key advantage of the system is its ability to combine:

- Browser-based interaction.
- AI-driven extraction reasoning.
- Context minimization.
- Traceability.
- Deterministic output generation.

Below are several practical real-world applications.

---

### 18.1 Enterprise Threat Intelligence and Security Monitoring

Large organizations continuously monitor:

- Cybersecurity advisories.
- Threat intelligence feeds.
- Vendor incident pages.
- Vulnerability disclosures.
- Operational status dashboards.
- Security blogs.
- Regulatory alerts.

Many of these sources are dynamic websites with changing layouts, client-side rendering, and inconsistent structures.

A browser-agent extraction pipeline like OmniBrief can:

- Monitor multiple security sources concurrently.
- Extract newly published incidents or vulnerabilities.
- Summarize operational risk updates.
- Generate internal security briefings automatically.
- Track changes across runs using trace files.
- Provide auditability for compliance teams.

Example enterprise workflow:

```text id="l2w0qe"
Security portals -> browser extraction -> AI summarization -> internal SOC briefing
```

This reduces manual analyst workload while preserving observability and traceability.

---

### 18.2 Financial and Market Intelligence Automation

Financial teams often collect information from:

- Market news websites.
- Earnings pages.
- Economic dashboards.
- Industry reports.
- Competitor announcements.
- Investor relations portals.

These sources frequently contain dynamic content and inconsistent layouts that are difficult to scrape reliably with traditional parsers.

The OmniBrief architecture can be adapted to:

- Monitor competitor updates.
- Track market-moving announcements.
- Generate executive financial briefings.
- Summarize earnings reports.
- Extract KPI changes from dashboards.
- Produce structured investment research summaries.

Example workflow:

```text id="z14mqm"
Financial websites -> browser agent extraction -> structured market summary -> executive report
```

The deterministic formatting layer is especially useful for standardized business reporting.

---

### 18.3 Internal Enterprise Knowledge Aggregation

Large organizations often have fragmented information spread across:

- Internal portals.
- Vendor dashboards.
- SaaS admin panels.
- Documentation systems.
- Operational monitoring pages.
- Team-specific reporting tools.

Employees spend significant time manually collecting updates from these systems.

The OmniBrief pattern can be extended into an internal enterprise intelligence layer that:

- Visits internal dashboards automatically.
- Extracts operational updates.
- Tracks important metrics.
- Produces daily summaries for leadership teams.
- Creates department-specific briefings.
- Maintains historical trace records for auditing.

Example workflow:

```text id="j9ghb1"
Internal portals -> AI browser extraction -> operational summary -> leadership dashboard
```

This becomes especially valuable in environments where information exists across multiple disconnected systems.

---

### Why These Use Cases Matter

These examples demonstrate that the value of the architecture is not only in web scraping itself.

The important engineering capability is:

```text id="1gh5ps"
Reliable extraction and summarization from dynamic, inconsistent, and semi-structured interfaces.
```

The combination of:

- Browser-based interaction.
- AI reasoning.
- Context minimization.
- Deterministic formatting.
- Telemetry and traceability.

makes the architecture suitable for production-oriented enterprise intelligence systems rather than only experimental scraping projects.