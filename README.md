# AI Code Reviewer & Optimizer (CLI)

A command‑line tool that automates pull‑request quality gates by combining rule‑based linters with AI‑powered code review suggestions. Designed for seamless integration into existing workflows (local CLI, CI pipelines, or desktop apps), it catches style, complexity, and security issues while providing contextual, human‑like recommendations.

---

## Table of Contents

- [Features & Capabilities](#features--capabilities)
- [Architecture & AI Model](#architecture--ai-model)
- [Technical Details](#technical-details)
  - [OpenAI API Integration](#openai-api-integration)
  - [Rule‑Based Tools](#rule-based-tools)
  - [CLI Design & Extensibility](#cli-design--extensibility)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Basic CLI](#basic-cli)
  - [CI Integration](#ci-integration)
  - [Output Formats](#output-formats)
- [Configuration](#configuration)
- [Billing & Cost Estimates](#billing--cost-estimates)
- [Development & Contribution](#development--contribution)
- [License](#license)

---

## Features & Capabilities

- **Automated Quality Gate**: Combines flake8, radon (cyclomatic complexity), and Bandit (security) checks with AI-driven suggestions.
- **AI Suggestions**: Uses OpenAI’s `gpt-4o-mini` model via ChatCompletion to provide human‑like feedback and confidence scoring.
- **Configurable Thresholds**: Tune rule‑based and AI confidence thresholds in `config.yaml`.
- **Multi‑File Support**: Accepts multiple diff files in one run; merges them into a unified report.
- **CI/CD & GitHub Action**: Easy integration for auto‑commenting on PRs and gating merges.
- **Output Options**: Console, Markdown file, JSON (future extension).
- **Guardrails**: Confidence cutoff and max‑comment limits to prevent noise.

---

## Architecture & AI Model

1. **Rule‑Based Engine**

   - **Flake8**: Naming conventions and style.
   - **Radon**: Cyclomatic complexity analysis.
   - **Bandit**: Security vulnerability detection.

2. **AI Review Engine**

   - **Model**: `gpt-4o-mini` (OpenAI ChatCompletion API).
   - **Prompt**: Custom prompt guiding the model to review diffs, suggest improvements, and end with `Confidence: X.YZ`.
   - **Guardrails**: Parses the final confidence line, includes suggestions only above `min_confidence`.

3. **Integration Layer**

   - **Python CLI**: Built with `click` for command‑line parsing.
   - **Post‑PR Comment**: Optional GitHub API integration via `requests`, using `GITHUB_TOKEN`.
   - **Diff Handling**: Reads one or more diff files, processes them in-memory, writes temporary files for linters.

---

## Technical Details

### OpenAI API Integration

- **Library**: Official `openai` Python SDK (v0.27+).
- **Endpoint**: `openai.ChatCompletion.create(...)`.
- **Authentication**: via `OPENAI_API_KEY` in `.env` or environment variable.
- **Model Selection**: `gpt-4o-mini` for a balance of speed, cost, and capability.
- **Error Handling**: Validates response structure, raises runtime errors on missing content or parse failures.

### Rule‑Based Tools

- **Flake8**: Invoked with `--diff` on a temporary diff file.
- **Radon (cc)**: Analyzes complexity, filters functions above threshold.
- **Bandit**: Run in text mode (`-f txt -q`), filters out JSON headers and progress logs.
- **Subprocess Isolation**: Each tool runs in its own process with capture of stdout.

### CLI Design & Extensibility

- **Click**: Modular commands and decorators.
- **Configurable**: `config.yaml` for tools, thresholds, AI settings, auto‑reject.
- **Plugin‑Ready**: Future support for additional linters or language adapters by extending `run_rule_checks`.

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/your-org/ai-code-reviewer.git
cd ai-code-reviewer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\Activate.ps1 # Windows PowerShell

# Install dependencies
pip install -e .
pip install radon bandit flake8

# Configure API key
echo "OPENAI_API_KEY=sk-..." > .env

# Customize settings
cp config.example.yaml config.yaml
# Edit config.yaml thresholds, AI settings
```

---

## Usage

### Basic CLI

```bash
# Single diff
python -m reviewer.cli path/to/changes.diff

# Multiple diffs + output file + auto-reject
python -m reviewer.cli diff1.diff diff2.diff \
  --auto-reject -o review_report.md
```

### CI Integration

Include in GitHub Actions: auto‑comment on PRs and optionally fail checks:

```yaml
- run: |
    git diff ${{ github.event.pull_request.base.sha }} ${{ github.sha }} -- '*.py' > pr.diff
    python -m reviewer.cli pr.diff -o pr_report.md --auto-reject
```

### Output Formats

- **Console**: Colorized Markdown.
- **File**: Markdown report via `-o` flag.
- **JSON**: (TBD) for dashboards.

---

## Configuration

See `config.yaml` for:

```yaml
rules:
  naming_convention:
    tool: flake8
    threshold: 5
  complexity:
    tool: radon
    threshold: 10
  security:
    tool: bandit
    threshold: 1

ai_review:
  temperature: 0.2
  max_comments: 10
  min_confidence: 0.7

auto_reject:
  enabled: false
  overall_threshold: 3
```

---

## Billing & Cost Estimates

- **OpenAI API**: \~200 tokens per PR review ⇒ \~\$0.0004 per review (@ \$0.002/1K tokens).
- **Linters**: Free open‑source.
- **CI/CD**: Free tier suffices for <2K monthly runs.
- **Infra**: <\$50/month if self‑hosting or using cloud functions.

**ROI**: Developers save \~10 min per PR ⇒ at \$60/hr ⇒ \$10 saved ⇒ break‑even in 1–2 PRs.

---

## Development & Contribution

- **Code style**: Black + Flake8. Run `black .` and `flake8 src/ tests/`.
- **Testing**: Pytest; see `tests/test_reviewer.py`.
- **Extending**: Add new tools in `run_rule_checks`, new output formats in `cli.py`.

Contributions welcome! Please open issues or PRs in GitHub.

---

## License

MIT © Hariram

