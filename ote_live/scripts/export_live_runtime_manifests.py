from __future__ import annotations

import argparse

from ote_live.models.registry import (
    DEFAULT_CANDIDATE_REGISTRY_PATH,
    DEFAULT_FEATURE_METADATA_PATH,
    DEFAULT_LONG_FEATURE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_BACKTEST_SUMMARY_PATH,
    DEFAULT_PREPARED_SUMMARY_PATH,
    DEFAULT_SHORT_FEATURE_PATH,
    DEFAULT_POLICY_ARTIFACT_DIR,
    build_direction_runtime_manifests,
    write_direction_runtime_manifests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Phase 0 live runtime manifests and policy packages into ote_live/runtime_manifests.",
    )
    parser.add_argument("--registry-path", default=str(DEFAULT_CANDIDATE_REGISTRY_PATH))
    parser.add_argument("--prepared-summary-path", default=str(DEFAULT_PREPARED_SUMMARY_PATH))
    parser.add_argument("--feature-metadata-path", default=str(DEFAULT_FEATURE_METADATA_PATH))
    parser.add_argument("--long-feature-path", default=str(DEFAULT_LONG_FEATURE_PATH))
    parser.add_argument("--short-feature-path", default=str(DEFAULT_SHORT_FEATURE_PATH))
    parser.add_argument("--policy-backtest-summary-path", default=str(DEFAULT_POLICY_BACKTEST_SUMMARY_PATH))
    parser.add_argument("--packaged-policy-dir", default=str(DEFAULT_POLICY_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    direction_manifests = build_direction_runtime_manifests(
        registry_path=args.registry_path,
        prepared_summary_path=args.prepared_summary_path,
        feature_metadata_path=args.feature_metadata_path,
        long_feature_path=args.long_feature_path,
        short_feature_path=args.short_feature_path,
        policy_backtest_summary_path=args.policy_backtest_summary_path,
        packaged_policy_dir=args.packaged_policy_dir,
    )
    output_dir = write_direction_runtime_manifests(direction_manifests, output_dir=args.output_dir)

    model_count = sum(len(manifest.models) for manifest in direction_manifests.values())
    for direction, manifest in direction_manifests.items():
        print(
            f"{direction}: {len(manifest.models)} models, "
            f"primary={manifest.recommendations.recommended_primary_model_id}, "
            f"strongest_v2={manifest.recommendations.recommended_strongest_v2_model_id}"
        )
    print(f"Wrote {model_count} model manifests and policies to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
