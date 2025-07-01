import os
import yaml
import subprocess
import tempfile
import shutil
import stat

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from reviewer.reviewer import run_rule_checks, run_ai_review

# Load config
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

app = FastAPI(title="AI Code Reviewer Service")


class ReviewRequest(BaseModel):
    diff: Optional[str] = None
    repo_url: Optional[str] = None
    pr_number: Optional[int] = None
    base: str = "main"


class ReviewResponse(BaseModel):
    naming_convention: List[str]
    complexity:       List[str]
    security:         List[str]
    ai_comments:      List[str]
    ai_score:         float


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    try:
        # 1) Compute unified diff text
        if req.repo_url and req.pr_number:
            tmp_dir = tempfile.mkdtemp(prefix="ai-review-")
            try:
                # a) shallow clone
                subprocess.run(
                    ["git", "clone", "--depth", "1", req.repo_url, tmp_dir],
                    check=True
                )
                # b) fetch & checkout PR branch
                subprocess.run(
                    ["git", "fetch", "origin", f"pull/{req.pr_number}/head:pr_branch"],
                    cwd=tmp_dir, check=True
                )
                subprocess.run(
                    ["git", "checkout", "pr_branch"],
                    cwd=tmp_dir, check=True
                )
                # c) get diff vs base
                diff_bytes = subprocess.check_output(
                    ["git", "diff", f"origin/{req.base}...HEAD"],
                    cwd=tmp_dir
                )
                diff_text = diff_bytes.decode("utf-8", errors="ignore")
            finally:
                def on_rm_error(func, path, exc_info):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                shutil.rmtree(tmp_dir, onerror=on_rm_error)

        else:
            diff_text = req.diff or ""

        # 2) Rule‐based checks
        rules = run_rule_checks(
            diff_text,
            cfg["rules"],
            repo_url=req.repo_url,
            pr_number=req.pr_number,
            base=req.base
        )

        # 3) AI‐based review
        ai_comments, ai_score = run_ai_review(diff_text, cfg["ai_review"])

        # 4) Return a flat JSON
        return ReviewResponse(
            naming_convention = rules.get("naming_convention", []),
            complexity        = rules.get("complexity", []),
            security          = rules.get("security", []),
            ai_comments       = ai_comments,
            ai_score          = ai_score
        )

    except subprocess.CalledProcessError as cpe:
        raise HTTPException(500, detail=f"Git error: {cpe}")
    except Exception as e:
        raise HTTPException(500, detail=str(e))
