"""Production Command-Line Interface for Retinal Disease Classification and Diagnostic Triage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessor import CLASS_LABELS, CLASS_DISPLAY_NAMES, RetinaPreprocessor
from src.retina_engine import RetinaEngine


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="retina-cli",
        description="Retinal Disease Computer-Aided Diagnostic (CAD) & Clinical Triage Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Multi-class retinal disease classification")
    pred_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input fundus image")
    pred_parser.add_argument("--clahe", action="store_true", help="Apply adaptive contrast enhancement (CLAHE)")
    pred_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # 2. uncertainty subcommand
    unc_parser = subparsers.add_parser("uncertainty", help="Monte Carlo Dropout uncertainty and entropy evaluation")
    unc_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input fundus image")
    unc_parser.add_argument("--passes", "-n", type=int, default=15, help="Number of Monte Carlo Dropout forward passes (default: 15)")
    unc_parser.add_argument("--clahe", action="store_true", help="Apply adaptive contrast enhancement (CLAHE)")
    unc_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # 3. explain subcommand
    exp_parser = subparsers.add_parser("explain", help="Grad-CAM visual pathology heatmap generation")
    exp_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input fundus image")
    exp_parser.add_argument("--target-class", "-c", type=str, default=None, help="Target disease class (cataract, diabetic_retinopathy, glaucoma, normal)")
    exp_parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity blend factor (0.1 - 0.9, default: 0.45)")
    exp_parser.add_argument("--colormap", type=str, default="jet", help="Colormap name (jet, turbo, viridis)")
    exp_parser.add_argument("--output", "-o", type=str, default=None, help="Filepath to save Grad-CAM overlay image")
    exp_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # 4. triage subcommand
    tri_parser = subparsers.add_parser("triage", help="Structured ophthalmological referral triage report")
    tri_parser.add_argument("--image", "-i", type=str, required=True, help="Path to input fundus image")
    tri_parser.add_argument("--clahe", action="store_true", help="Apply adaptive contrast enhancement (CLAHE)")
    tri_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # 5. batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch cohort triage from a directory of fundus images")
    batch_parser.add_argument("--input", "-i", type=str, required=True, help="Directory containing fundus images")
    batch_parser.add_argument("--output", "-o", type=str, default=None, help="Path to save triage report (CSV or JSON)")
    batch_parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to process")
    batch_parser.add_argument("--clahe", action="store_true", help="Apply adaptive contrast enhancement (CLAHE)")
    batch_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # 6. benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run automated evaluation and calibration benchmark")
    bench_parser.add_argument("--report", "-r", type=str, default=None, help="Path to save markdown benchmark report")
    bench_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    return parser


def handle_predict(args: argparse.Namespace) -> int:
    """Execute predict command."""
    img_path = Path(args.image)
    if not img_path.exists():
        sys.stderr.write(f"Error: Retinal image not found at {img_path}\n")
        return 1

    engine = RetinaEngine()
    res = engine.analyze_image(img_path, run_mc_dropout=False, run_gradcam=False, apply_clahe=args.clahe)

    if args.json:
        payload = {
            "image_path": str(img_path),
            "predicted_condition": res["predicted_display_name"],
            "predicted_class": res["predicted_class"],
            "confidence_percentage": res["confidence_percentage"],
            "probabilities": res["deterministic_probabilities"],
            "image_quality": res["image_quality"],
            "urgency_tier": res["clinical_triage"]["urgency_tier"],
            "response_sla": res["clinical_triage"]["response_sla"],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 64)
    print("  RETINAL CAD DIAGNOSTIC PREDICTION")
    print("=" * 64)
    print(f"Exam Image:             {img_path.name}")
    print(f"Optical Quality:        Grade {res['image_quality']['grade']} ({res['image_quality']['status']})")
    print(f"Resolution:             {res['image_quality']['resolution'][0]}x{res['image_quality']['resolution'][1]}")
    print("-" * 64)
    print(f"Primary Diagnosis:      {res['predicted_display_name'].upper()}")
    print(f"Confidence:             {res['confidence_percentage']:.2f}%")
    print("-" * 64)
    print("Probability Distribution:")
    for cls_name, prob in res["deterministic_probabilities"].items():
        disp = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
        bar = "#" * int(prob / 5)
        print(f"  {disp:<24} {prob:>6.2f}% | {bar}")
    print("-" * 64)
    print(f"Triage Urgency:         {res['clinical_triage']['urgency_tier']}")
    print(f"Response SLA Target:    {res['clinical_triage']['response_sla']}")
    print("=" * 64 + "\n")
    return 0


def handle_uncertainty(args: argparse.Namespace) -> int:
    """Execute uncertainty command."""
    img_path = Path(args.image)
    if not img_path.exists():
        sys.stderr.write(f"Error: Retinal image not found at {img_path}\n")
        return 1

    engine = RetinaEngine()
    res = engine.analyze_image(
        img_path,
        run_mc_dropout=True,
        run_gradcam=False,
        apply_clahe=args.clahe,
        num_mc_samples=args.passes,
    )
    unc = res["uncertainty"]

    if args.json:
        payload = {
            "image_path": str(img_path),
            "mc_samples_count": unc["num_mc_samples"],
            "predicted_condition": unc["predicted_display_name"],
            "confidence_percentage": unc["confidence_percentage"],
            "epistemic_uncertainty_std": unc["epistemic_uncertainty_std"],
            "shannon_entropy_bits": unc["shannon_entropy_bits"],
            "normalized_entropy": unc["normalized_entropy"],
            "mutual_information": unc["mutual_information"],
            "uncertainty_tier": unc["uncertainty_tier"],
            "reliability_score": unc["reliability_score"],
            "second_opinion_recommended": unc["second_opinion_recommended"],
            "class_breakdown": unc["class_breakdown"],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 68)
    print("  MONTE CARLO DROPOUT UNCERTAINTY & ENTROPY ANALYSIS")
    print("=" * 68)
    print(f"Exam Image:              {img_path.name}")
    print(f"Stochastic Passes:       {unc['num_mc_samples']} iterations (training=True)")
    print(f"Primary Condition:       {unc['predicted_display_name']}")
    print(f"Confidence (Mean):       {unc['confidence_percentage']:.2f}%")
    print(f"Epistemic Uncertainty:   {unc['epistemic_uncertainty_std']:.4f} (std deviation)")
    print(f"Shannon Predictive Entropy: {unc['shannon_entropy_bits']:.4f} bits (Norm: {unc['normalized_entropy']:.2f})")
    print(f"Mutual Information:      {unc['mutual_information']:.4f}")
    print("-" * 68)
    print(f"Uncertainty Classification: {unc['uncertainty_tier']}")
    print(f"Diagnostic Reliability:     {unc['reliability_score']:.1f} / 100")
    opinion_flag = "[FLAGGED] Mandatory" if unc["second_opinion_recommended"] else "[CLEARED] Routine Protocol"
    print(f"Second Specialist Opinion:  {opinion_flag}")
    print("-" * 68)
    print("Per-Class Posterior Estimates (95% Credible Intervals):")
    for cls_name, meta in unc["class_breakdown"].items():
        print(f"  {meta['class_name']:<22} Mean: {meta['mean_percentage']:>5.1f}% | Std: {meta['epistemic_std']:.3f} | CI95: [{meta['ci_95_low']*100:.1f}%, {meta['ci_95_high']*100:.1f}%]")
    print("=" * 68 + "\n")
    return 0


def handle_explain(args: argparse.Namespace) -> int:
    """Execute explain command."""
    img_path = Path(args.image)
    if not img_path.exists():
        sys.stderr.write(f"Error: Retinal image not found at {img_path}\n")
        return 1

    engine = RetinaEngine(gradcam_colormap=args.colormap)
    res = engine.analyze_image(
        img_path,
        run_mc_dropout=False,
        run_gradcam=True,
        gradcam_target_class=args.target_class,
    )
    cam = res["gradcam"]

    out_file = args.output or f"gradcam_{img_path.stem}.png"
    out_p = Path(out_file)
    cam["overlay_pil"].save(out_p)

    if args.json:
        payload = {
            "image_path": str(img_path),
            "target_class": cam["target_class_display"],
            "blend_alpha": cam["blend_alpha"],
            "saved_overlay_path": str(out_p.resolve()),
            "focal_pathology_boxes": cam["focal_pathology_boxes"],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 68)
    print("  GRAD-CAM VISUAL PATHOLOGY EXPLAINABILITY")
    print("=" * 68)
    print(f"Exam Image:              {img_path.name}")
    print(f"Target Feature Class:    {cam['target_class_display']}")
    print(f"Convolutional Layer:     top_conv (EfficientNetB3 (None, 7, 7, 1536))")
    print(f"Colormap / Opacity:      {args.colormap} / {cam['blend_alpha']:.2f}")
    print(f"Saved Overlay Image:     {out_p.resolve()}")
    print("-" * 68)
    boxes = cam["focal_pathology_boxes"]
    if boxes:
        print(f"Detected Focal Regions:  {len(boxes)} primary pathology cluster(s)")
        for idx, b in enumerate(boxes, 1):
            print(f"  Cluster {idx}: Peak Intensity={b['peak_intensity']:.3f} | Center={b['center_point']} | Normalized Box={b['normalized_bbox']}")
    else:
        print("Detected Focal Regions:  Diffuse activation across retina (no localized hotspot)")
    print("=" * 68 + "\n")
    return 0


def handle_triage(args: argparse.Namespace) -> int:
    """Execute triage command."""
    img_path = Path(args.image)
    if not img_path.exists():
        sys.stderr.write(f"Error: Retinal image not found at {img_path}\n")
        return 1

    engine = RetinaEngine()
    res = engine.analyze_image(img_path, run_mc_dropout=True, run_gradcam=False, apply_clahe=args.clahe)
    tri = res["clinical_triage"]

    if args.json:
        print(json.dumps(tri, indent=2))
        return 0

    print("\n" + "=" * 70)
    print("  OPHTHALMOLOGICAL REFERRAL & CLINICAL TRIAGE PROTOCOL")
    print("=" * 70)
    print(f"Patient Exam:            {img_path.name}")
    print(f"CAD Indication:          {tri['display_name']} ({tri['confidence_percentage']:.1f}% Confidence)")
    print(f"ICD-10 Diagnostic Code:  {tri['icd10_code']}")
    print("-" * 70)
    print(f"Referral Urgency:        {tri['urgency_tier']}")
    print(f"Response SLA Target:     {tri['response_sla']}")
    opinion_badge = "[TRIGGERED] Mandatory Second Specialist Review" if tri["second_opinion_required"] else "[CLEAR] Standard Physician Sign-Off"
    print(f"Second-Opinion Trigger:  {opinion_badge}")
    print(f"\nClinical Rationale:\n  {tri['clinical_rationale']}")
    print("-" * 70)
    print("Recommended Diagnostic Modalities:")
    for mod in tri["recommended_modalities"]:
        print(f"  * {mod}")
    print("-" * 70)
    print("Frontline Clinical Action Checklist:")
    for idx, act in enumerate(tri["action_checklist"], 1):
        print(f"  {idx}. {act}")
    print("-" * 70)
    print(f"Notice: {tri['regulatory_disclaimer']}")
    print("=" * 70 + "\n")
    return 0


def handle_batch(args: argparse.Namespace) -> int:
    """Execute batch command."""
    in_path = Path(args.input)
    if not in_path.exists():
        sys.stderr.write(f"Error: Input directory or file not found at {in_path}\n")
        return 1

    if in_path.is_dir():
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        image_files = sorted([p for p in in_path.iterdir() if p.suffix.lower() in extensions])
    else:
        # Single image or text list
        image_files = [in_path]

    if not image_files:
        sys.stderr.write(f"Error: No image files found in {in_path}\n")
        return 1

    if args.limit:
        image_files = image_files[: int(args.limit)]

    engine = RetinaEngine()
    batch_res = engine.analyze_batch(image_files, run_mc_dropout=True, run_gradcam=False)
    summary_df = batch_res["triage_summary_table"]

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if str(out_p).endswith(".json"):
            summary_df.to_json(out_p, orient="records", indent=2)
        else:
            summary_df.to_csv(out_p, index=False)

    if args.json:
        payload = {
            "total_exams_evaluated": batch_res["total_exams_evaluated"],
            "urgent_cases_count": batch_res["urgent_cases_count"],
            "triage_summary": summary_df.to_dict(orient="records"),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 70)
    print("  BATCH RETINAL COHORT TRIAGE AUDIT")
    print("=" * 70)
    print(f"Total Exams Evaluated:   {batch_res['total_exams_evaluated']}")
    print(f"Urgent / Flagged Cases:  {batch_res['urgent_cases_count']} patient(s)")
    print("-" * 70)
    print(f"{'Exam ID':<20} {'Condition':<22} {'Conf':<7} {'Urgency SLA':<16} {'Flag'}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        flag = "[!]" if row["Second_Opinion_Required"] else "[OK]"
        print(f"{row['Patient_Identifier'][:18]:<20} {row['Predicted_Condition'][:20]:<22} {row['Confidence_Pct']:>5.1f}% {row['SLA_Target'][:15]:<16} {flag}")
    if args.output:
        print(f"\nSaved full triage ledger to: {args.output}")
    print("=" * 70 + "\n")
    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark evaluation suite."""
    from evals.runner import run_eval_suite

    rep_path = Path(args.report) if args.report else None
    res = run_eval_suite(report_path=rep_path)

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    disc = res["classification_metrics"]
    calib = res["calibration_metrics"]
    lat = res["latency_benchmark"]

    print("\n" + "=" * 68)
    print("  RETINAL CAD BENCHMARK & CLINICAL CALIBRATION SUMMARY")
    print("=" * 68)
    print(f"Evaluated Test Cases:    {res['num_samples']}")
    print(f"Top-1 Diagnostic Acc:    {disc['accuracy'] * 100:.2f}%")
    print(f"Macro-Averaged F1:       {disc['macro_f1']:.4f}")
    print(f"Macro-Averaged Recall:   {disc['macro_recall']:.4f}")
    print(f"Multi-Class Brier Loss:  {calib['multi_class_brier']:.4f}")
    print(f"Expected Calib Error:    {calib['ece']:.4f}")
    print("-" * 68)
    print(f"Mean Inference Latency:  {lat['mean_latency_ms']:.2f} ms / image")
    print(f"Single-Core Throughput:  {lat['throughput_images_per_sec']:.1f} images / sec")
    print("=" * 68 + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "predict":
        return handle_predict(args)
    elif args.subcommand == "uncertainty":
        return handle_uncertainty(args)
    elif args.subcommand == "explain":
        return handle_explain(args)
    elif args.subcommand == "triage":
        return handle_triage(args)
    elif args.subcommand == "batch":
        return handle_batch(args)
    elif args.subcommand == "benchmark":
        return handle_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
