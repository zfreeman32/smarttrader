from __future__ import annotations

import argparse

from ote_live.env import env_path, load_repo_env
from ote_live.policies.packager import (
    DEFAULT_POLICY_ARTIFACT_DIR,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_THRESHOLD_POLICY_ROOT,
    package_candidate_policy_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    load_repo_env()
    parser = argparse.ArgumentParser(
        description="Package model-specific threshold/abstain policy artifacts into ote_live/policy_artifacts.",
    )
    parser.add_argument(
        "--threshold-policy-root",
        default=str(env_path("OTE_LIVE_THRESHOLD_POLICY_ROOT", DEFAULT_THRESHOLD_POLICY_ROOT)),
    )
    parser.add_argument("--output-dir", default=str(env_path("OTE_LIVE_POLICY_ARTIFACT_DIR", DEFAULT_POLICY_ARTIFACT_DIR)))
    parser.add_argument("--registry-path", default=str(env_path("OTE_LIVE_REGISTRY_PATH", DEFAULT_REGISTRY_PATH)))
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    return parser


def main() -> int:
    load_repo_env()
    parser = build_parser()
    args = parser.parse_args()

    written_paths = package_candidate_policy_artifacts(
        threshold_policy_root=args.threshold_policy_root,
        output_dir=args.output_dir,
        registry_path=args.registry_path,
        model_ids=args.model_ids,
    )
    for path in written_paths:
        print(path)
    print(f"Wrote {len(written_paths)} packaged policy artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
