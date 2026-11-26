"""Check the evidence and outcomes for PS4 submission conventions."""

import base64
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
            "additions": 0, "deletions": 0,
        })
        files.append({
            "filename": "analyses/member_a.ipynb", "status": "added",
            "patch": '@@ -0,0 +1 @@\n+{"cells":[{"cell_type":"code","source":["def member_a(): pass"]}]}',
            "additions": 1, "deletions": 0,
        })

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(acceptance, "merge_evidence_files", lambda _: [
        {"filename": "analyses/merge_only.py"},
    ])
    monkeypatch.setattr(acceptance, "github", lambda _: {"merge_base_commit": {"sha": "merge-base"}})

    def source(path, ref, *, limit=24_000, source_required=False):
        if path == "analyses/legacy.py" and limit is None:
            assert source_required
            return {"merge-base": "def prior_function():", "head": "def revised_function():"}[ref]
        return "source at merge"

    monkeypatch.setattr(acceptance, "evidence_file", source)

    def review(_service, _url, _key, payload):
        evidence = json.loads(payload["input"])
        assert evidence["title"] == title
        assert evidence["description"] == description
        changes = {change["path"]: change for change in evidence["code_changes"]}
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
            assert "def member_a()" in changes["analyses/member_a.ipynb"]["patch"]
            assert {"path": "analyses/member_a.ipynb", "status": "added", "previous_path": None} in evidence[
                "submitted_files"
            ]
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


def test_oversized_evidence_combines_late_findings_automatically(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / ".github" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    acceptance = importlib.import_module("ps4_acceptance")
    budget = 4_000
    monkeypatch.setattr(acceptance, "RUBRIC_INPUT_BUDGET", budget)
    label = "I am Member A."
    module = "analyses/learning_curve.py"
    evidence = {
        "criteria": [{"id": "PS4-S4", "requirement": "Use descriptive names and avoid internal labels."}],
        "title": "Plot LightGBM learning curves", "assignment_source": "course",
        "exercise_sources": [], "earlier_review_sources": [],
        "description": "Fit the constrained model. " * 450 + label,
        "changed_paths": [module],
        "submitted_files": [{"path": module, "status": "added", "previous_path": None}],
        "code_changes": [{"path": module, "status": "added", "previous_path": None,
                          "evidence_complete": True, "patch": "+def plot_learning_curve():"}],
        "snapshots": {module: "def plot_learning_curve():"},
    }
    requests = []
    schema = {"properties": {"items": {"items": {"properties": {"id": {"enum": ["PS4-S4"]}},
                                                  "required": ["id", "status", "evidence"]}}}}
    request = {"instructions": "Check the rubric.", "text": {"format": {"schema": schema}}}

    def review(payload):
        assert len(payload["input"].encode("utf-8")) <= budget
        data = json.loads(payload["input"])
        requests.append(data)
        if "observations" in data:
            assert "unless another observation specifically disproves them" in payload["instructions"]
            facts = {item["fact"] for item in data["observations"]}
            assert {label, "Descriptive name shown."} <= facts
            return [{"id": "PS4-S4", "status": "missing", "evidence": label}]
        texts = [record.get("text") or "" for record in data["records"]]
        facts = ["Descriptive name shown."]
        if any(label in text for text in texts):
            facts.append(label)
        return [{"id": "PS4-S4", "status": "needs_review", "evidence": "Batched evidence.",
                 "observations": [{"subject": "PR description", "fact": fact} for fact in facts]}]

    monkeypatch.setattr(acceptance, "review_evidence", review)
    items = acceptance.review_submission(request, evidence)

    assert sum("records" in request for request in requests) > 1
    assert any("observations" in request for request in requests)
    assert items == [{"id": "PS4-S4", "status": "missing", "evidence": label}]

    def review_with_verdict_only(payload):
        data = json.loads(payload["input"])
        if "records" in data and any(label in (record.get("text") or "") for record in data["records"]):
            return [{"id": "PS4-S4", "status": "missing", "evidence": label, "observations": []}]
        return review(payload)

    monkeypatch.setattr(acceptance, "review_evidence", review_with_verdict_only)
    assert acceptance.review_submission(request, evidence)[0]["status"] == "missing"

    def omit_requirement(payload):
        if any(label in (record.get("text") or "") for record in json.loads(payload["input"])["records"]):
            return []
        return review(payload)

    monkeypatch.setattr(acceptance, "review_evidence", omit_requirement)
    with pytest.raises(RuntimeError, match="each requirement exactly once"):
        acceptance.review_submission(request, evidence)

    def review_many_facts(payload):
        data = json.loads(payload["input"])
        assert len(payload["input"].encode("utf-8")) <= budget
        if "records" in data:
            if len(data["records"]) > 5:
                raise acceptance.RubricOutputLimit("Output exhausted in test.")
            facts = [{"subject": "same file", "fact": str(record.get("offset")) + "x" * 100}
                     for record in data["records"] if record["kind"] == "PR description"]
        elif "This is an intermediate pass." in payload["instructions"]:
            facts = [{"subject": "same file", "fact": "Complementary evidence from this shard."}]
        else:
            assert data["observations"] == [{"id": "PS4-S4", "subject": "same file",
                                              "fact": "Complementary evidence from this shard."}]
            return [{"id": "PS4-S4", "status": "met", "evidence": "Evidence reconciled."}]
        return [{"id": "PS4-S4", "status": "needs_review", "evidence": "Partial evidence.",
                 "observations": facts}]

    monkeypatch.setattr(acceptance, "review_evidence", review_many_facts)
    assert acceptance.review_submission(request, evidence)[0]["status"] == "met"

    def review_without_compression(payload):
        data = json.loads(payload["input"])
        assert len(payload["input"].encode("utf-8")) <= budget
        if "records" in data:
            return review_many_facts(payload)
        if "This is an intermediate pass." in payload["instructions"]:
            facts = [{"subject": record["subject"], "fact": record["fact"]} for record in data["observations"]]
            return [{"id": "PS4-S4", "status": "needs_review", "evidence": "Distinct facts.", "observations": facts}]
        if data.get("subject_verdicts"):
            assert all(item["evidence"] == "Earlier evidence considered." for item in data["subject_verdicts"])
        return [{"id": "PS4-S4", "status": "met", "evidence": "Earlier evidence considered."}]

    monkeypatch.setattr(acceptance, "review_evidence", review_without_compression)
    assert acceptance.review_submission(request, evidence)[0]["status"] == "met"

    def review_related_subjects(data, _instructions, _payload):
        if "subject_verdicts" in data:
            assert {item["subject"] for item in data["subject_verdicts"]} == {"runner", "model"}
            return [{"id": "PS4-S3", "status": "met", "evidence": "Runner calls the separate model."}]
        subject = data["observations"][0]["subject"]
        return [{"id": "PS4-S3", "status": "missing" if subject == "runner" else "met", "evidence": subject}]

    facts = [{"subject": "runner", "fact": "Calls the model."}, {"subject": "model", "fact": "Implements the exercise."}]
    combined = acceptance.reconcile_in_order(
        {"criteria": [{"id": "PS4-S3"}]}, [{"observations": facts}], request, "", review_related_subjects,
    )
    assert combined["status"] == "met"


def test_long_patch_lines_keep_their_diff_direction(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / ".github" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    acceptance = importlib.import_module("ps4_acceptance")
    hunk = "@@ -1 +1 @@"
    patch = f"{hunk}\n-{'x' * 9_000}removed_name\n+{'y' * 9_000}added_name"

    records = list(acceptance.patch_records({"path": "analysis.ipynb"}, patch))
    removed = next(record for record in records if "removed_name" in record["text"])
    added = next(record for record in records if "added_name" in record["text"])

    assert (removed["side"], removed["patch_line"], removed["hunk"]) == ("deletion", 2, hunk)
    assert (added["side"], added["patch_line"], added["hunk"]) == ("addition", 3, hunk)


def test_reads_blob_when_contents_api_omits_large_file_content(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / ".github" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    acceptance = importlib.import_module("ps4_acceptance")
    source = "# large notebook\n" * 80_000
    responses = {
        "/contents/analysis.ipynb?ref=base": {"type": "file", "encoding": "none", "sha": "blob"},
        "/git/blobs/blob": {"encoding": "base64", "content": base64.b64encode(source.encode()).decode()},
    }
    def github(path, *, accept=None):
        if path.startswith("/contents/"):
            assert accept == "application/vnd.github.object+json"
        return responses[path]

    monkeypatch.setattr(acceptance, "github", github)

    assert acceptance.evidence_file("analysis.ipynb", "base", limit=None) == source


def test_omitted_rename_patch_uses_the_extensionless_old_source(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / ".github" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    acceptance = importlib.import_module("ps4_acceptance")
    source = "# coding: latin-1\n# régression\ndef legacy_member_a(): pass"
    blob = {"type": "file", "encoding": "base64", "content": base64.b64encode(source.encode("latin-1")).decode()}
    responses = {
        "/compare/base...head": {"merge_base_commit": {"sha": "fork"}},
        "/contents/legacy?ref=fork": blob,
        "/contents/analysis.py?ref=head": blob,
    }
    monkeypatch.setattr(acceptance, "github", lambda path, **_: responses[path])

    changes = acceptance.submission_code_changes(
        {"base": {"sha": "base"}, "head": {"sha": "head"}},
        [{"filename": "analysis.py", "previous_filename": "legacy", "status": "renamed",
          "additions": 0, "deletions": 0}],
    )
    assert changes[0]["evidence_complete"]
    assert changes[0]["base_source"] == changes[0]["head_source"] == source
