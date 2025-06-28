# src/reviewer/reviewer.py

import os, subprocess, openai, requests
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_rule_checks(diff: str, rules_cfg: dict) -> dict:
    # write diff to temp file
    with open("temp.diff", "w") as f: f.write(diff)
    results = {}
    for name, cfg in rules_cfg.items():
        if cfg["tool"] == "flake8":
            cmd = ["flake8", "--diff", "temp.diff"]
        elif cfg["tool"] == "radon":
            cmd = ["radon", "cc", "--min", "A", "temp.diff"]
        elif cfg["tool"] == "bandit":
            cmd = ["bandit", "-r", ".", "-f", "json"]
        else:
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True)
        issues = proc.stdout.splitlines()[: cfg["threshold"]]
        results[name] = issues
    return results

def run_ai_review(diff: str, ai_cfg: dict):
    prompt = (
        "You are a senior code reviewer. Give concise suggestions, "
        "then end with a line ‘Confidence: X.YZ’ where X.YZ ∈ [0,1].\n\n"
        f"{diff}"
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=ai_cfg["temperature"],
    )
    # 1) Extract the first choice
    choice = resp.choices[0]
     # 2) Get the message object (may be None in types)
    message = getattr(choice, "message", None)
    if message is None or message.content is None:
        raise RuntimeError("No content returned from OpenAI API")
    # 3) Now it's safe to strip and split
    full_text = message.content.strip()
    lines = full_text.splitlines()
    if not lines:
        raise RuntimeError("OpenAI returned empty content")
    # 4) Separate comments vs. confidence line
    *comments, conf_line = lines
    # 5) Parse the confidence value (after the colon)
    try:
        score = float(conf_line.split(":", 1)[1].strip())
    except (IndexError, ValueError):
        raise RuntimeError(f"Unable to parse confidence from: {conf_line!r}")

    return comments, score

def post_pr_comment(body: str):
    token = os.getenv("GITHUB_TOKEN")
    pr = os.getenv("GITHUB_PR_NUMBER")
    repo = os.getenv("GITHUB_REPOSITORY")
    if token and pr and repo:
        url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
        headers = {"Authorization": f"token {token}"}
        requests.post(url, json={"body": body}, headers=headers)
    else:
        print(body)
