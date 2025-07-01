import os
import sys
import subprocess
import tempfile
import shutil
import stat

import click
import yaml

from reviewer.reviewer import run_rule_checks, run_ai_review, post_pr_comment

@click.command()
@click.option("--repo-url", help="HTTPS URL of the GitHub repo")
@click.option("--pr-number", type=int, help="Pull request number")
@click.option("--base-branch", "base", default="main", help="Base branch to diff against")
@click.argument("diff_file", required=False, type=click.Path(exists=True))
@click.option("--auto-reject/--no-auto-reject", default=False, help="Fail exit if violations exceed threshold")
@click.option("-o", "--output-file", type=click.Path(), help="Write report to this file")
def main(repo_url, pr_number, base, diff_file, auto_reject, output_file):
    """AI Code Reviewer CLI."""
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    # 1) Determine the unified diff text
    if repo_url and pr_number:
        # PR-mode: shallow clone & checkout, then git-diff
        tmp_dir = tempfile.mkdtemp(prefix="ai-review-")
        try:
            # a) clone
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, tmp_dir],
                check=True
            )
            # b) fetch + checkout PR branch
            subprocess.run(
                ["git", "fetch", "origin", f"pull/{pr_number}/head:pr_branch"],
                cwd=tmp_dir, check=True
            )
            subprocess.run(
                ["git", "checkout", "pr_branch"],
                cwd=tmp_dir, check=True
            )
            # c) get unified diff vs base branch
            diff_bytes = subprocess.check_output(
                ["git", "diff", f"origin/{base}...HEAD"],
                cwd=tmp_dir
            )
            diff = diff_bytes.decode("utf-8", errors="ignore")
        finally:
            # cleanup (handle read-only on Windows)
            def on_rm_error(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(tmp_dir, onerror=on_rm_error)

    elif diff_file:
        # Diff-file mode
        diff = open(diff_file, encoding="utf-8").read()

    else:
        click.echo("❌ Provide either a DIFF_FILE or both --repo-url and --pr-number", err=True)
        sys.exit(1)

    # 2) Rule-based checks
    rules = run_rule_checks(diff, cfg["rules"], repo_url, pr_number, base)

    # 3) AI-based review (same diff)
    ai_comments, ai_score = run_ai_review(diff, cfg["ai_review"])

    # 4) Build the markdown report
    total = sum(len(v) for v in rules.values()) + len(ai_comments)
    lines = ["# AI Code Quality Report", ""]
    lines.append("## Rule-based Violations")
    for name, items in rules.items():
        lines.append(f"### {name} ({len(items)})")
        lines += [f"- {i}" for i in items]

    lines.append("\n## AI Suggestions")
    if ai_score >= cfg["ai_review"]["min_confidence"]:
        lines += [f"- {c}" for c in ai_comments[: cfg["ai_review"]["max_comments"]]]
    else:
        lines.append(f"*Skipped AI feedback (confidence {ai_score:.2f} < threshold).*")

    report = "\n".join(lines)
    click.echo(report)

    # 5) Optional file write
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        click.echo(f"✅ Report written to {output_file}")

    # 6) Post to GitHub in CI
    if os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_PR_NUMBER") and os.getenv("GITHUB_REPOSITORY"):
        post_pr_comment(report)

    # 7) Auto-reject
    if auto_reject or cfg["auto_reject"]["enabled"]:
        if total > cfg["auto_reject"]["overall_threshold"]:
            click.echo(f"{total} violations > threshold", err=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
