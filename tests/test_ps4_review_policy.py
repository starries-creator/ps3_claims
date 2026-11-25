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
    description = "Plots train and test deviance." if compliant else "I am Member A."
    runner_diff = (
        "+from analyses.learning_curve import plot_learning_curve"
        if compliant else "+def fit_and_plot_entire_exercise(frame):"
    )
    runner = {
        "filename": "analyses/ps3_script.py", "status": "modified", "patch": runner_diff,
    }
    files = [runner]
    if compliant:
        files.append({
            "filename": "analyses/learning_curve.py",
            "status": "renamed",
            "previous_filename": "analyses/member_a.py",
            "patch": "+def plot_learning_curve(frame):",
        })

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(acceptance, "merge_evidence_files", lambda _: files)
    monkeypatch.setattr(acceptance, "evidence_file", lambda *_: "source at merge")

    def review(_service, _url, _key, payload):
        evidence = json.loads(payload["input"])
        assert evidence["title"] == title
        assert evidence["description"] == description
        changes = {change["path"]: change for change in evidence["python_changes"]}
        assert changes["analyses/ps3_script.py"]["patch"] == runner_diff
        if compliant:
            module = changes["analyses/learning_curve.py"]
            assert module["status"] == "renamed"
            assert module["previous_path"] == "analyses/member_a.py"
            assert module["patch"] == "+def plot_learning_curve(frame):"
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
    pr = {"title": title, "body": description, "number": 30}
    if compliant:
        acceptance.assess(rubric, ["2"], pr, "test-merge", set())
    else:
        with pytest.raises(SystemExit, match="rubric items are missing"):
            acceptance.assess(rubric, ["2"], pr, "test-merge", set())

    result = summary.read_text()
    expected = "met" if compliant else "missing"
    assert f"| PS4-S3 | {expected} |" in result
    assert f"| PS4-S4 | {expected} |" in result
    assert "##[error]" not in result
