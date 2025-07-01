import os
import subprocess
import tempfile
import shutil
import json
import stat
import openai
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def run_rule_checks(
    diff: str,
    rules_cfg: dict,
    repo_url: Optional[str] = None,
    pr_number: Optional[int] = None,
    base: str = "main",
) -> dict:
    """
    If repo_url+pr_number are provided, clone the PR and run full-file flake8, radon, bandit.
    Otherwise, run flake8 --diff against the raw diff and skip the others.
    """
    if repo_url and pr_number:
        return _clone_and_lint(repo_url, pr_number, base, rules_cfg)
    else:
        return _diff_only_lint(diff, rules_cfg)


def _diff_only_lint(diff: str, rules_cfg: dict) -> dict:
    # Write diff to temp file
    with open("temp.diff", "w", encoding="utf-8") as f:
        f.write(diff)

    results = {}
    for name, cfg in rules_cfg.items():
        issues = []
        if cfg["tool"] == "flake8":
            proc = subprocess.run(
                ["flake8", "--diff", "temp.diff"],
                capture_output=True,
                text=True,
            )
            issues = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        # radon & bandit are skipped in diff-only mode
        results[name] = issues[: cfg.get("threshold", 0)] if cfg.get("threshold") else issues
    return results


def _clone_and_lint(repo_url, pr_number, base, rules_cfg):
    """
    Shallow-clone the repo, checkout the PR branch, diff against base,
    then run flake8, radon, and bandit on the changed files.
    """
    temp_dir = tempfile.mkdtemp(prefix="ai-review-")
    results = {name: [] for name in rules_cfg}

    def git(cmd):
        return subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True, check=True)

    try:
        # 1) Clone & fetch PR
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True)
        git(["git", "fetch", "origin", f"pull/{pr_number}/head:pr_branch"])
        git(["git", "checkout", "pr_branch"])

        # 2) Get changed Python files
        diff_stdout = git(
            ["git", "diff", f"origin/{base}...HEAD", "--name-only", "--", "*.py"]
        ).stdout
        files = [f for f in diff_stdout.splitlines() if f.endswith(".py")]

        # 3) For each rule, run the appropriate tool
        for name, cfg in rules_cfg.items():
            tool = cfg["tool"]
            thresh = cfg.get("threshold", 0)
            issues = []

            if tool == "flake8":
                for f in files:
                    p = subprocess.run(
                        ["flake8", f],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                    )
                    issues += [l.strip() for l in p.stdout.splitlines() if l.strip()]

            elif tool == "radon":
                for f in files:
                    p = subprocess.run(
                        ["radon", "cc", "--min", "A", f],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                    )
                    issues += [l.strip() for l in p.stdout.splitlines() if l.strip()]

            elif tool == "bandit":
                # scan entire PR tree for security
                p = subprocess.run(
                    ["bandit", "-r", ".", "-f", "json", "-q"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                )
                try:
                    data = json.loads(p.stdout)
                    for item in data.get("results", []):
                        fname = item.get("filename")
                        txt = item.get("issue_text")
                        issues.append(f"{fname}: {txt}")
                except json.JSONDecodeError:
                    pass

            # Apply threshold
            if thresh and len(issues) > thresh:
                issues = issues[:thresh]

            results[name] = issues

        return results

    finally:
        # Clean up even if Windows marks files read-only
        def on_rm_error(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(temp_dir, onerror=on_rm_error)


def run_ai_review(diff: str, ai_cfg: dict):
    prompt = (
        "You are a senior code reviewer. Give concise suggestions, "
        "then end with a line 'Confidence: X.YZ' where X.YZ ∈ [0,1].\n\n"
        f"{diff}"
    )
    resp = openai.chat.completions.create(
        model=ai_cfg.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=ai_cfg.get("temperature", 0.2),
    )
    choice = resp.choices[0]
    message = getattr(choice, "message", None)
    if not message or not message.content:
        raise RuntimeError("No content returned from OpenAI API")

    lines = message.content.strip().splitlines()
    *comments, conf = lines
    try:
        score = float(conf.split(":", 1)[1].strip())
    except:
        raise RuntimeError(f"Failed to parse confidence from {conf!r}")

    return comments, score


def post_pr_comment(body: str):
    token = os.getenv("GITHUB_TOKEN")
    pr = os.getenv("GITHUB_PR_NUMBER")
    repo = os.getenv("GITHUB_REPOSITORY")
    if token and pr and repo:
        url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
        requests.post(url, json={"body": body}, headers={"Authorization": f"token {token}"})
