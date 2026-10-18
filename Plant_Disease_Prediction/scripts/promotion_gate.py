"""Conservative, read-only promotion gate. Never changes or promotes model files."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess:
    print("\n$", " ".join(command))
    return subprocess.run(command, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", default="trained_plant_disease_model.keras")
    parser.add_argument("--candidate", default="trained_plant_disease_model_improved.keras")
    parser.add_argument("--validation", default="New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid")
    parser.add_argument("--external", default=r"D:\FYP\External_Dataset\Test_OOD")
    parser.add_argument("--mapping", default="class_names.json")
    parser.add_argument("--labels", default="label_mapping.yaml")
    parser.add_argument("--output", default="reports/promotion_gate")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prod = str(Path(args.production))
    candidate = str(Path(args.candidate))
    validation = str(Path(args.validation))
    external = str(Path(args.external))
    mapping = str(Path(args.mapping))
    labels = str(Path(args.labels))
    py = sys.executable

    checks = []
    def add(name: str, passed: bool, evidence: str):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    test = run([py, "-m", "pytest", "tests/test_preprocess.py", "-q"])
    add("Preprocessing tests", test.returncode == 0, f"pytest exit code {test.returncode}")

    mapping_status = []
    for label, model_path in (("production", prod), ("candidate", candidate)):
        report_path = f"reports/promotion_gate/class_mapping_{label}.md"
        result = run([py, "scripts/verify_class_mapping.py", "--model", model_path,
                      "--dataset", validation, "--mapping", mapping, "--output", report_path])
        mapping_status.append(result.returncode == 0)
    add("Class mapping check (both checkpoints)", all(mapping_status),
        f"production={'PASS' if mapping_status[0] else 'FAIL'}, candidate={'PASS' if mapping_status[1] else 'FAIL'}")

    metric_results = {}
    for dataset_label, dataset_path, map_path in (
        ("validation", validation, labels),
        ("external", external, labels),
    ):
        for model_label, model_path in (("production", prod), ("candidate", candidate)):
            result_dir = f"reports/promotion_gate/{dataset_label}_{model_label}"
            result = run([py, "evaluate.py", "--model", model_path, "--dataset", dataset_path,
                          "--mapping", map_path, "--class-names", mapping, "--output", result_dir])
            metrics_path = Path(result_dir) / "metrics.json"
            if result.returncode != 0 or not metrics_path.is_file():
                metric_results[(dataset_label, model_label)] = None
            else:
                metric_results[(dataset_label, model_label)] = json.loads(metrics_path.read_text(encoding="utf-8"))

    pv = metric_results.get(("validation", "production"))
    cv = metric_results.get(("validation", "candidate"))
    val_pass = bool(pv and cv and pv.get("accuracy") is not None and cv.get("accuracy") is not None and cv["accuracy"] >= pv["accuracy"])
    add("Candidate validation accuracy >= production", val_pass,
        f"production={pv.get('accuracy') if pv else 'N/A'}, candidate={cv.get('accuracy') if cv else 'N/A'}")

    pe = metric_results.get(("external", "production"))
    ce = metric_results.get(("external", "candidate"))
    ext_pass = bool(pe and ce and pe.get("mapped_class_accuracy") is not None and ce.get("mapped_class_accuracy") is not None and ce["mapped_class_accuracy"] >= pe["mapped_class_accuracy"])
    add("Candidate external mapped-class accuracy >= production", ext_pass,
        f"production={pe.get('mapped_class_accuracy') if pe else 'N/A'}, candidate={ce.get('mapped_class_accuracy') if ce else 'N/A'}")

    ece_values = {}
    for model_label, model_path in (("production", prod), ("candidate", candidate)):
        cal_dir = f"reports/promotion_gate/calibration_{model_label}"
        result = run([py, "calibration.py", "--model", model_path, "--validation", validation,
                      "--mapping", mapping, "--output", cal_dir])
        metric_path = Path(cal_dir) / "calibration_metrics.json"
        ece_values[model_label] = (
            json.loads(metric_path.read_text(encoding="utf-8")).get("ece_after")
            if result.returncode == 0 and metric_path.is_file() else None
        )
    ece_pass = (ece_values["production"] is not None and ece_values["candidate"] is not None
                and ece_values["candidate"] <= ece_values["production"])
    add("Candidate held-out validation ECE <= production", ece_pass,
        f"production={ece_values['production']}, candidate={ece_values['candidate']}")

    passed = all(item["status"] == "PASS" for item in checks)
    lines = ["# Model promotion gate", "", f"**Overall: {'PASS' if passed else 'FAIL'}**", "",
             "| Gate | Status | Evidence |", "|---|---|---|"]
    lines.extend(f"| {item['check']} | {item['status']} | {item['evidence']} |" for item in checks)
    lines.extend(["", "This gate is read-only: it never copies, renames, or promotes a model. A missing dataset, mapping, dependency, or report is a FAIL, not a pass.", ""])
    (output / "promotion_gate.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
