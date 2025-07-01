# AI Code Reviewer & Optimizer

This repository provides a unified toolset for automatic Pull Request (PR) quality gating, combining rule-based linters (Flake8, Radon, Bandit) with an AI-driven review via OpenAI's ChatCompletion API. It consists of:

1. **CLI Application** (`cli.py`) for local/diff-based or repo/PR-based scanning.
2. **Python Library** (`reviewer.py`) encapsulating rule-check and AI-review logic.
3. **FastAPI Service** (`service.py`) exposing a REST endpoint for external integrations (e.g., WPF frontend).
4. **Configuration** (`config.yaml`) driving thresholds, AI parameters, and auto-reject policies.

---

## Table of Contents

- [Architecture](#architecture)
- [Components](#components)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [REST API](#rest-api)
- [Integration with WPF](#integration-with-wpf)
- [Security & Guardrails](#security--guardrails)
- [Deployment & Packaging](#deployment--packaging)
- [Future Work](#future-work)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
flowchart LR
    subgraph Local
      CLI[CLI Entry Point]\n      ReviewLib[reviewer.py]
    end

    subgraph ServiceHost
      API[FastAPI Service]\n      ReviewLib2[reviewer.py]
    end

    CLI --> ReviewLib --> {Rule Check}
    CLI --> ReviewLib --> {AI Review}
    API --> ReviewLib2 --> {Rule Check}
    API --> ReviewLib2 --> {AI Review}

    subgraph Rule Check
      Flake8
      Radon
      Bandit
    end

    subgraph AI Review
      OpenAI[ChatCompletion API]
    end
```

1. **CLI** and **Service** share the same core module (`reviewer.py`).
2. **Rule Check** forks into three subprocess calls:
   - **Flake8** for style/naming violations.
   - **Radon** for cyclomatic complexity grades.
   - **Bandit** for security issue scanning.
3. **AI Review** calls the OpenAI API with the raw unified diff to produce human‐like suggestions plus a confidence score.

---

## Components

### 1. `reviewer.py`

- **Entry functions**:
  - `run_rule_checks(diff, rules_cfg, repo_url=None, pr_number=None, base="main")`
  - `run_ai_review(diff, ai_cfg)`
  - `post_pr_comment(body)`
- **Modes**:
  - **Diff‐Only**: writes `diff` to `temp.diff`, runs `flake8 --diff`.
  - **PR‐Mode**: shallow clones the target repo, fetches `refs/pull/<N>/head`, checks out a temp branch, diffs against `base`, runs full‐file linters.

### 2. `cli.py`

- Based on **Click** for argument parsing.
- Supports:
  - `--repo-url` + `--pr-number` + `--base-branch` → PR‐Mode.
  - `diff_file` argument → Diff‐Only Mode.
  - `-o/--output-file` to save markdown report.
  - `--auto-reject` to exit non‐zero if total violations exceed threshold.

### 3. `service.py`

- **FastAPI** app exposing ``.
- Accepts JSON `{ diff?, repo_url?, pr_number?, base? }`.
- Returns flat JSON:
  ```json
  {
    "naming_convention": [...],
    "complexity": [...],
    "security": [...],
    "ai_comments": [...],
    "ai_score": 0.87
  }
  ```

### 4. `config.yaml`

```yaml
rules:
  naming_convention:
    tool: flake8
    threshold: 0      # 0 = no limit
  complexity:
    tool: radon
    threshold: 0
  security:
    tool: bandit
    threshold: 0

a i_review:
  temperature: 0.2
  max_comments: 10
  min_confidence: 0.7

auto_reject:
  enabled: false
  overall_threshold: 0
```

- **Thresholds** cap per‐tool findings (0 means unlimited).
- **AI parameters** tune GPT temperature and comment count.
- **Auto‐reject** can enforce CI gate based on `overall_threshold`.

---

## Data Flow

1. **Input**: Diff text (from CLI file or GitHub PR).
2. **Rule Check**:
   - Identify changed files (via `git diff --name-only`).
   - Spawn subprocesses for each tool.
   - Collect and threshold results.
3. **AI Review**:
   - Send unified diff in prompt.
   - Parse out suggestions and confidence.
4. **Output**:
   - Markdown report (CLI).
   - JSON payload (Service).

---

## CLI Usage

```bash
# Diff‐only mode:
python -m reviewer.cli examples/sample_pr.diff -o report.md

# PR‐Mode:
python -m reviewer.cli \
  --repo-url https://github.com/YourOrg/Repo.git \
  --pr-number 5 \
  --base-branch main \
  -o report_pr.md
```

Report structure:

```markdown
# AI Code Quality Report

## Rule-based Violations
### naming_convention (N)
- file:line:code  description

### complexity (M)
- file:func …

### security (K)
- file: issue_text

## AI Suggestions
- suggestion 1
- suggestion 2
```

---

## REST API

```http
POST /review HTTP/1.1
Content-Type: application/json

{
  "repo_url": "https://github.com/YourOrg/Repo.git",
  "pr_number": 5,
  "base": "main"
}
```

Response:

```json
{
  "naming_convention": [...],
  "complexity": [...],
  "security": [...],
  "ai_comments": [...],
  "ai_score": 0.85
}
```

---

## Integration with WPF

1. **HTTP client** calls `POST /review`.
2. **Deserialize** JSON into a C# model:
   ```csharp
   public class ReviewResult {
     public List<string> NamingConvention { get; set; }
     public List<string> Complexity { get; set; }
     public List<string> Security { get; set; }
     public List<string> AiComments  { get; set; }
     public double AiScore { get; set; }
   }
   ```
3. **Bind** lists to UI controls (ListView, DataGrid).
4. **Display** AI comments and a visual pass/fail if `AiScore >= min_confidence`.

---

## Security & Guardrails

- **Read-only temp cleanup** handles Windows file locks via `os.chmod`.
- **Timeouts/Circuit Breakers** can be added around OpenAI calls and subprocesses.
- **Error handling** returns HTTP 500 with details; CI-mode posts comments only when `GITHUB_TOKEN` is set.

---

## Deployment & Packaging

- **Dependencies** in `requirements.txt`:
  ```
  fastapi
  uvicorn
  click
  pyyaml
  python-dotenv
  openai
  flake8
  radon
  bandit
  ```
- **Publish** as a PyPI package via `pyproject.toml` metadata.
- **Docker** image can wrap Uvicorn service for Kubernetes.

---

## Future Work

- Multi-language support (Java, JavaScript) via language-specific linters.
- In‐browser VSCode extension for inline diff reviews.
- Slack/MS Teams bot for PR notifications.
- Automated PR comments with GitHub Actions integration.

---

## Troubleshooting

- ``** fails**: verify repo URL and PR number.
- **Empty arrays** in diff-only mode: ensure your patch adds lint violations.
- **AI parser errors**: confirm the model returns a `Confidence: X.YZ` line.

---
