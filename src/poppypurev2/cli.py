"""Command-line entry point for project checks and later ML workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poppypurev2.config import DEFAULT_CONFIG, load_config
from poppypurev2.records import make_record, write_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poppypure")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Validate configuration and show data paths")
    smoke = commands.add_parser("smoke", help="Write a Phase 0 run record")
    smoke.add_argument("--output", type=Path, required=True)
    download = commands.add_parser("acquire", help="Download and verify source data")
    download.add_argument(
        "--external-root",
        type=Path,
        default=Path("/Volumes/Backup Plus/PoppyPurev2"),
    )
    download.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    audit = commands.add_parser(
        "audit", help="Inventory images and generate Phase 1 reports"
    )
    audit.add_argument("--record", type=Path)
    commands.add_parser("source-metadata", help="Snapshot public source provenance")
    commands.add_parser("freeze-data", help="Freeze a reviewed Phase 1 manifest")
    protocol = commands.add_parser(
        "freeze-evaluation", help="Create or verify Phase 2 splits"
    )
    protocol.add_argument("--record", type=Path)
    protocol.add_argument("--version", choices=("v1", "v2", "v3", "v4"))
    preprocessing = commands.add_parser(
        "preprocess-qc", help="Training-only Phase 3 preprocessing review"
    )
    preprocessing.add_argument("--record", type=Path)
    masks = commands.add_parser(
        "mask-qc", help="Training-only background-mask prototypes"
    )
    masks.add_argument("--record", type=Path)
    features = commands.add_parser(
        "extract-features", help="Training-only classical features"
    )
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--record", type=Path)
    features.add_argument(
        "--partition", choices=("train", "validation"), default="train"
    )
    controls = commands.add_parser(
        "prepare-controls", help="Train/validation diagnostic inputs"
    )
    controls.add_argument("--output", type=Path, required=True)
    baseline = commands.add_parser(
        "train-baselines", help="Run bounded Phase 4 comparisons"
    )
    baseline.add_argument("--output", type=Path, required=True)
    plots = commands.add_parser("plot-phase4", help="Phase 4 loss/result figures")
    plots.add_argument("--output", type=Path, required=True)
    deep = commands.add_parser("train-deep", help="Bounded Phase 5 ResNet comparisons")
    deep.add_argument("--output", type=Path, required=True)
    commands.add_parser(
        "refit-final", help="Fixed 466-image Phase 6 refit and pretest seal"
    )
    final_test = commands.add_parser(
        "evaluate-final", help="Score the sealed final model once"
    )
    final_test.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    commands.add_parser("report-final", help="Report saved final-test predictions")
    predict = commands.add_parser(
        "predict", help="Predict a by-volume class from an image"
    )
    predict.add_argument("image", type=Path)
    predict.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_CONFIG.parent.parent / "models/poppypure-v1",
    )
    predict.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    predict.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    # Inference is independent of workspace configs, raw data and pretrained files.
    if args.command == "predict":
        from poppypurev2.evaluation import encoded
        from poppypurev2.inference import predict_image

        try:
            if args.output and args.output.exists():
                raise ValueError(f"Prediction output already exists: {args.output}")
            result = predict_image(args.model_dir, args.image, args.device)
            if args.output:
                args.output.write_bytes(encoded(result))
            print(json.dumps(result, indent=2))
            return 0
        except (OSError, ValueError) as error:
            parser.error(str(error))

    config, config_sha256 = load_config(args.config)
    project_root = args.config.resolve().parent.parent
    if args.command == "refit-final":
        from poppypurev2.final import refit_final

        print(json.dumps(refit_final(project_root), indent=2))
    elif args.command == "evaluate-final":
        from poppypurev2.final import evaluate_final

        print(json.dumps(evaluate_final(project_root, args.device), indent=2))
    elif args.command == "report-final":
        from poppypurev2.final_reports import report_final

        print(json.dumps(report_final(project_root), indent=2))
    elif args.command == "train-deep":
        from poppypurev2.deep import run_phase5
        from poppypurev2.deep_acceptance import verify_phase5
        from poppypurev2.deep_reports import render_phase5

        print(json.dumps(run_phase5(project_root, args.output), indent=2))
        print(json.dumps(verify_phase5(project_root, args.output), indent=2))
        print(json.dumps(render_phase5(project_root, args.output), indent=2))
    elif args.command == "plot-phase4":
        from poppypurev2.plots import phase4_plots

        print(json.dumps(phase4_plots(project_root, args.output), indent=2))
    elif args.command == "prepare-controls":
        from poppypurev2.controls import prepare_controls

        print(json.dumps(prepare_controls(project_root, args.output), indent=2))
    elif args.command == "train-baselines":
        from poppypurev2.baselines import run_phase4

        print(json.dumps(run_phase4(project_root, args.output), indent=2))
    elif args.command in {"mask-qc", "extract-features"}:
        if args.command == "mask-qc":
            from poppypurev2.masks import mask_review

            summary = mask_review(project_root)
        else:
            from poppypurev2.features import extract_partition

            summary = extract_partition(project_root, args.output, args.partition)
        if args.record:
            record = make_record(
                kind="phase3_" + args.command.replace("-", "_"),
                project_root=project_root,
                config_sha256=config_sha256,
                seed=config["project"]["seed"],
                status="completed",
                dataset_manifest_sha256=json.loads(
                    (project_root / "data/dataset_freeze.json").read_text()
                )["manifest_sha256"],
                split_sha256=summary["split_sha256"],
                preprocessing=summary,
            )
            write_record(record, args.record)
        print(json.dumps(summary, indent=2))
    elif args.command == "preprocess-qc":
        from poppypurev2.preprocessing import training_qc

        summary = training_qc(project_root)
        if args.record:
            record = make_record(
                kind="phase3_preprocessing_qc",
                project_root=project_root,
                config_sha256=config_sha256,
                seed=config["project"]["seed"],
                status="candidates_processed",
                dataset_manifest_sha256=json.loads(
                    (project_root / "data/dataset_freeze.json").read_text()
                )["manifest_sha256"],
                split_sha256=summary["split_sha256"],
                preprocessing=summary,
            )
            write_record(record, args.record)
        print(json.dumps(summary, indent=2))
    elif args.command == "freeze-evaluation":
        from poppypurev2.evaluation import freeze_protocol

        version = args.version or Path(config["evaluation"]["protocol"]).parent.name
        frozen = freeze_protocol(project_root, version, config["project"]["seed"])
        if args.record:
            record = make_record(
                kind="phase2_evaluation_protocol",
                project_root=project_root,
                config_sha256=config_sha256,
                seed=config["project"]["seed"],
                status="frozen_limited_scope",
                dataset_manifest_sha256=frozen["manifest_sha256"],
                split_sha256=frozen["artifacts"]["assignments.csv"],
            )
            record["evaluation_freeze"] = frozen
            write_record(record, args.record)
        print(json.dumps(frozen, indent=2))
    elif args.command == "source-metadata":
        from poppypurev2.provenance import snapshot_sources

        snapshot_sources(project_root)
    elif args.command == "acquire":
        from poppypurev2.acquisition import acquire

        print(
            json.dumps(
                acquire(project_root, args.external_root, args.workers), indent=2
            )
        )
    elif args.command == "freeze-data":
        from poppypurev2.audit import freeze_dataset

        print(json.dumps(freeze_dataset(project_root), indent=2))
    elif args.command == "audit":
        from poppypurev2.audit import run_audit

        summary = run_audit(project_root)
        if args.record:
            record = make_record(
                kind="phase1_audit",
                project_root=project_root,
                config_sha256=config_sha256,
                seed=config["project"]["seed"],
                status="reviewed"
                if not (
                    summary["decision_counts"].get("pending_review", 0)
                    or summary["near_duplicate_reviews"].get("pending_review", 0)
                )
                else "needs_review",
                dataset_manifest_sha256=summary["manifest_sha256"],
            )
            record["source_archive_sha256"] = summary["archive_sha256"]
            record["audit_summary"] = summary
            write_record(record, args.record)
        print(json.dumps(summary, indent=2))
    if args.command == "doctor":
        paths = {
            name: str(project_root / value) for name, value in config["paths"].items()
        }
        print(
            json.dumps(
                {
                    "project": config["project"]["name"],
                    "classes_volume_percent": config["project"][
                        "classes_volume_percent"
                    ],
                    "config_sha256": config_sha256,
                    "paths": paths,
                },
                indent=2,
            )
        )
    elif args.command == "smoke":
        record = make_record(
            kind="phase0_smoke",
            project_root=project_root,
            config_sha256=config_sha256,
            seed=config["project"]["seed"],
            status="passed",
        )
        write_record(record, args.output)
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
