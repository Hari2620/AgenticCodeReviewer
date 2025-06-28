# tests/test_reviewer.py

from src.reviewer.reviewer import run_ai_review

def test_ai_confidence_parsing():
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-print('hi')\n+print('hello')\n"
    comments, score = run_ai_review(diff, {"temperature": 0})
    assert isinstance(comments, list)
    assert 0.0 <= score <= 1.0
