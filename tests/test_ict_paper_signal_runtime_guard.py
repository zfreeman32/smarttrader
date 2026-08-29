from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ote_live.scripts.run_es_live_collector import (
    _validate_ict_paper_signal_runtime,
)
from scripts.audit_ict_paper_signal_readiness import (
    BUNDLE_ID,
    DEFAULT_BUNDLE_DIR,
    EXPECTED_MODEL_IDS,
)


def _args(**overrides):
    values = {
        "include_ict": True,
        "ict_long_runtime_manifest_path": (
            "ote_live/runtime_manifests/ict_es_paper_signal_20260813/"
            "live_runtime_manifest_long.json"
        ),
        "ict_short_runtime_manifest_path": (
            "ote_live/runtime_manifests/ict_es_paper_signal_20260813/"
            "live_runtime_manifest_short.json"
        ),
        "all_models_active": False,
        "ict_paper_signal_trial_enabled": True,
        "ibkr_enabled": True,
        "ibkr_account_mode": "paper",
        "ibkr_port": 4002,
        "ibkr_allow_delayed_fallback": False,
        "heartbeat_file": "ote_live/runtime_data/health/test-heartbeat.json",
        "allow_ict_clean_handoff": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ict_paper_signal_runtime_accepts_scoped_paper_configuration() -> None:
    _validate_ict_paper_signal_runtime(_args(), readiness_audit=_ready_audit)


def test_ict_paper_signal_runtime_passes_explicit_clean_handoff_only_when_requested() -> None:
    observed: list[bool] = []

    def capture_audit(**kwargs):
        observed.append(bool(kwargs["allow_clean_stopped_handoff"]))
        return _ready_audit()

    _validate_ict_paper_signal_runtime(
        _args(allow_ict_clean_handoff=True),
        readiness_audit=capture_audit,
    )
    _validate_ict_paper_signal_runtime(
        _args(allow_ict_clean_handoff=False),
        readiness_audit=capture_audit,
    )

    assert observed == [True, False]


def test_ict_paper_signal_runtime_rejects_all_models_active_override() -> None:
    with pytest.raises(ValueError, match="ALL_MODELS_ACTIVE=false"):
        _validate_ict_paper_signal_runtime(_args(all_models_active=True))


def test_ict_paper_signal_runtime_rejects_disabled_trial_switch() -> None:
    with pytest.raises(ValueError, match="launch switch is disabled"):
        _validate_ict_paper_signal_runtime(
            _args(ict_paper_signal_trial_enabled=False)
        )


def test_ict_paper_signal_runtime_rejects_nonpaper_account_mode() -> None:
    with pytest.raises(ValueError, match="IBKR_ACCOUNT_MODE=paper"):
        _validate_ict_paper_signal_runtime(_args(ibkr_account_mode="live"))


def test_ict_paper_signal_runtime_rejects_live_or_custom_endpoint() -> None:
    with pytest.raises(ValueError, match="paper endpoint"):
        _validate_ict_paper_signal_runtime(_args(ibkr_port=4001))


def test_ict_paper_signal_runtime_rejects_disabled_ibkr() -> None:
    with pytest.raises(ValueError, match="IBKR_ENABLED=true"):
        _validate_ict_paper_signal_runtime(_args(ibkr_enabled=False))


def test_ict_paper_signal_runtime_rejects_delayed_fallback() -> None:
    with pytest.raises(ValueError, match="IBKR_ALLOW_DELAYED_FALLBACK=false"):
        _validate_ict_paper_signal_runtime(
            _args(ibkr_allow_delayed_fallback=True)
        )


@pytest.mark.parametrize(
    ("field", "renamed_path"),
    (
        (
            "ict_long_runtime_manifest_path",
            "ote_live/runtime_manifests/renamed/live_runtime_manifest_long.json",
        ),
        (
            "ict_short_runtime_manifest_path",
            "ote_live/runtime_manifests/renamed/live_runtime_manifest_short.json",
        ),
    ),
)
def test_ict_paper_signal_runtime_rejects_renamed_or_copied_bundle_paths(
    field: str,
    renamed_path: str,
) -> None:
    with pytest.raises(ValueError, match="locked to the exact controlled bundle"):
        _validate_ict_paper_signal_runtime(_args(**{field: renamed_path}))


def test_ict_paper_signal_runtime_requires_full_ready_contract() -> None:
    def blocked_audit(**_kwargs):
        return {
            "bundle_id": BUNDLE_ID,
            "status": "blocked",
            "ready_to_start": False,
            "blocking_reasons": [
                {"code": "historical_feature_parity_not_passed", "message": "blocked"}
            ],
            "facts": {
                "bundle_dir": str(DEFAULT_BUNDLE_DIR.resolve()),
                "model_ids": sorted(EXPECTED_MODEL_IDS),
            },
        }

    with pytest.raises(ValueError, match="historical_feature_parity_not_passed"):
        _validate_ict_paper_signal_runtime(
            _args(),
            readiness_audit=blocked_audit,
        )


def test_ict_paper_signal_runtime_rejects_audit_identity_mismatch() -> None:
    def wrong_bundle_audit(**_kwargs):
        result = _ready_audit()
        result["bundle_id"] = "renamed_bundle"
        return result

    with pytest.raises(ValueError, match="bundle identity/content mismatch"):
        _validate_ict_paper_signal_runtime(
            _args(),
            readiness_audit=wrong_bundle_audit,
        )


def test_excluding_ict_skips_controlled_bundle_preflight() -> None:
    def unexpected_audit(**_kwargs):
        raise AssertionError("readiness audit must not run when ICT is excluded")

    _validate_ict_paper_signal_runtime(
        _args(include_ict=False),
        readiness_audit=unexpected_audit,
    )


def _ready_audit(**_kwargs):
    return {
        "bundle_id": BUNDLE_ID,
        "status": "ready_to_start",
        "ready_to_start": True,
        "blocking_reasons": [],
        "facts": {
            "bundle_dir": str(DEFAULT_BUNDLE_DIR.resolve()),
            "model_ids": sorted(EXPECTED_MODEL_IDS),
        },
    }
