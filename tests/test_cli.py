"""Unit tests for the Retina CAD CLI."""

import json
from pathlib import Path
import pytest

from cli import build_parser, main

SAMPLE_IMG = Path(__file__).resolve().parent.parent / "data" / "sample_fundus" / "cataract.png"


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "retina-cli" in out or "usage:" in out.lower()


def test_cli_empty_args(capsys: pytest.CaptureFixture) -> None:
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()


def test_cli_predict_text(capsys: pytest.CaptureFixture) -> None:
    code = main(["predict", "--image", str(SAMPLE_IMG)])
    assert code == 0
    out = capsys.readouterr().out
    assert "RETINAL CAD DIAGNOSTIC PREDICTION" in out
    assert "Primary Diagnosis:" in out
    assert "Confidence:" in out


def test_cli_predict_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["predict", "--image", str(SAMPLE_IMG), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "predicted_condition" in data
    assert "confidence_percentage" in data
    assert "probabilities" in data


def test_cli_uncertainty_text(capsys: pytest.CaptureFixture) -> None:
    code = main(["uncertainty", "--image", str(SAMPLE_IMG), "--passes", "3"])
    assert code == 0
    out = capsys.readouterr().out
    assert "MONTE CARLO DROPOUT UNCERTAINTY" in out
    assert "Shannon Predictive Entropy:" in out


def test_cli_uncertainty_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["uncertainty", "--image", str(SAMPLE_IMG), "--passes", "3", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "mc_samples_count" in data
    assert "shannon_entropy_bits" in data
    assert "uncertainty_tier" in data


def test_cli_explain(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out_cam = tmp_path / "test_cam.png"
    code = main(["explain", "--image", str(SAMPLE_IMG), "--output", str(out_cam)])
    assert code == 0
    assert out_cam.exists()
    out = capsys.readouterr().out
    assert "GRAD-CAM VISUAL PATHOLOGY" in out


def test_cli_triage(capsys: pytest.CaptureFixture) -> None:
    code = main(["triage", "--image", str(SAMPLE_IMG)])
    assert code == 0
    out = capsys.readouterr().out
    assert "CLINICAL TRIAGE PROTOCOL" in out
    assert "ICD-10 Diagnostic Code:" in out
    assert "Referral Urgency:" in out


def test_cli_triage_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["triage", "--image", str(SAMPLE_IMG), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "urgency_tier" in data
    assert "icd10_code" in data


def test_cli_batch_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["batch", "--input", str(SAMPLE_IMG.parent), "--limit", "2", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "total_exams_evaluated" in data
    assert "triage_summary" in data


def test_cli_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rep = tmp_path / "cli_bench.md"
    code = main(["benchmark", "--report", str(rep)])
    assert code == 0
    assert rep.exists()
    out = capsys.readouterr().out
    assert "RETINAL CAD BENCHMARK" in out


def test_cli_benchmark_json(capsys: pytest.CaptureFixture) -> None:
    code = main(["benchmark", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "classification_metrics" in data
    assert "calibration_metrics" in data


def test_cli_missing_image(capsys: pytest.CaptureFixture) -> None:
    code = main(["predict", "--image", "non_existent_file.png"])
    assert code == 1


def test_cli_missing_batch_dir(capsys: pytest.CaptureFixture) -> None:
    code = main(["batch", "--input", "non_existent_directory/"])
    assert code == 1

