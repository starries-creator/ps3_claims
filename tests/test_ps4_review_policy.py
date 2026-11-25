"""Check the evidence and outcomes for PS4 submission conventions."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("compliant", [True, False])
def test_submission_rules_use_pr_description_and_python_changes(
    compliant, monkeypatch, tmp_path
):
    scripts = Path(__file__).resolve().parents[1] / ".github" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    acceptance = importlib.import_module("ps4_acceptance")
    rubric = acceptance.load_rubric()
    title = "PS4 Ex2 — Plot the LightGBM learning curve"
    description = "Plots train and test deviance. " * 300
    if not compliant:
        description += "I am Member A."
    runner_line = (
        "+from analyses.learning_curve import plot_learning_curve"
        if compliant else "+def fit_and_plot_entire_exercise(frame):"
    )
    runner_diff = f"@@ -40,0 +41 @@\n{runner_line}"
    module_diff = (
        "@@ -1 +1,2 @@\n-def member_a(frame):\n+    # " + "x" * 8_001
        + "\n+def plot_learning_curve(frame):"
    )
    runner = {
        "filename": "analyses/ps3_script.py", "status": "modified", "patch": runner_diff,
        "additions": 1, "deletions": 0,
    }
    files = [runner]
    if compliant:
        files.append({
            "filename": "analyses/learning_curve.py",
            "status": "renamed",
            "previous_filename": "analyses/member_a.py",
            "patch": module_diff,
            "additions": 2,
            "deletions": 1,
        })
    else:
        files.append({
            "filename": "analyses/legacy.py", "status": "modified",
            "additions": 1, "deletions": 1,
        })

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(acceptance, "merge_evidence_files", lambda _: [
        {"filename": "analyses/merge_only.py"},
    ])
    monkeypatch.setattr(acceptance, "github", lambda _: {"merge_base_commit": {"sha": "merge-base"}})

    def source(path, ref, *, limit=24_000):
        if path == "analyses/legacy.py" and limit is None:
            return {"merge-base": "def prior_function():", "head": "def revised_function():"}[ref]
        return "source at merge"

    monkeypatch.setattr(acceptance, "evidence_file", source)

    def review(_service, _url, _key, payload):
        evidence = json.loads(payload["input"])
        assert evidence["title"] == title
        assert evidence["description"] == description
        changes = {change["path"]: change for change in evidence["python_changes"]}
        assert changes["analyses/ps3_script.py"]["patch"] == runner_diff
        assert changes["analyses/ps3_script.py"]["evidence_complete"]
        assert "analyses/merge_only.py" not in changes
        assert "analyses/merge_only.py" in evidence["snapshots"]
        if compliant:
            module = changes["analyses/learning_curve.py"]
            assert module["status"] == "renamed"
            assert module["previous_path"] == "analyses/member_a.py"
            assert module["patch"] == module_diff
            assert module["evidence_complete"]
        else:
            module = changes["analyses/legacy.py"]
            assert module["patch"] is None
            assert module["base_source"] == "def prior_function():"
            assert module["head_source"] == "def revised_function():"
            assert module["evidence_complete"]
        assert {"PS4-S3", "PS4-S4"} <= {item["id"] for item in evidence["criteria"]}
        assert ("analyses/learning_curve.py" in evidence["snapshots"]) == compliant
        items = [
            {
                "id": item["id"],
                "status": (
                    "missing" if not compliant and item["id"] in {"PS4-S3", "PS4-S4"}
                    else "met"
                ),
                "evidence": "Test evidence containing ##[error] in an untrusted label.",
            }
            for item in evidence["criteria"]
        ]
        return {
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps({"items": items})}
            ]}],
        }

    monkeypatch.setattr(acceptance, "request_json", review)
    pr = {"title": title, "body": description, "number": 30,
          "base": {"sha": "base"}, "head": {"sha": "head"}}
    if compliant:
        acceptance.assess(rubric, ["2"], pr, "test-merge", files)
    else:
        with pytest.raises(SystemExit, match="rubric items are missing"):
            acceptance.assess(rubric, ["2"], pr, "test-merge", files)

    result = summary.read_text()
    expected = "met" if compliant else "missing"
    assert f"| PS4-S3 | {expected} |" in result
    assert f"| PS4-S4 | {expected} |" in result
    assert "##[error]" not in result
