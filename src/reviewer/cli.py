# src/reviewer/cli.py

import click, yaml, sys
from .reviewer import run_rule_checks, run_ai_review, post_pr_comment

@click.command()
@click.argument("diff_file", type=click.Path(exists=True))
@click.option("--auto-reject/--no-auto-reject", default=False)
def main(diff_file, auto_reject):
    """AI Code Reviewer CLI."""
    cfg = yaml.safe_load(open("config.yaml"))
    diff = open(diff_file).read()

    # 1) Rule-based checks
    rule_violations = run_rule_checks(diff, cfg["rules"])

    # 2) AI-based review
    ai_comments, ai_score = run_ai_review(diff, cfg["ai_review"])

    # 3) Aggregate into report
    total = sum(len(v) for v in rule_violations.values()) + len(ai_comments)
    lines = ["# AI Code Quality Report", ""]

    lines.append("## Rule-based Violations")
    for name, issues in rule_violations.items():
        lines.append(f"### {name} ({len(issues)})")
        lines += [f"- {i}" for i in issues]

    lines.append("\n## AI Suggestions")
    if ai_score >= cfg["ai_review"]["min_confidence"]:
        lines += [f"- {c}" for c in ai_comments[: cfg["ai_review"]["max_comments"]]]
    else:
        lines.append(f"*Skipped AI feedback (confidence {ai_score:.2f} < threshold).*")

    report = "\n".join(lines)
    click.echo(report)

    # 4) Post to PR
    post_pr_comment(report)

    # 5) Auto-reject if configured
    if auto_reject or cfg["auto_reject"]["enabled"]:
        if total > cfg["auto_reject"]["overall_threshold"]:
            click.echo(f"{total} violations > threshold", err=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
