from __future__ import annotations

from typing import Dict

from model_testing.ote_abstain_policy import HardAbstainConfig
from model_testing.ote_threshold_policy import ThresholdSearchConfig
from models.ote_registry_loader import OTEModelRecord


TARGETED_FILTER_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "full_run_v2": {
        "long_ote_champion_v1": {
            "abstain_composite_regimes": ("strong_up_high",),
            "minimum_probability_quantile": 0.20,
        },
        "short_ote_candidate_tcn_v2": {},
    },
    "ict_es_primary_prune_v1": {
        "ict_long_continuation_xgb_v1": {
            "abstain_session_regimes": ("london", "new_york"),
            "abstain_composite_regimes": (
                "ranging_high",
                "strong_down_medium",
                "strong_up_medium",
            ),
            "apply_to_base_policy_variants": True,
        },
        "ict_long_meta_xgb_v1": {
            "abstain_composite_regimes": ("strong_down_high",),
            "apply_to_base_policy_variants": True,
        },
        "ict_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "strong_down_low",
                "strong_up_low",
            ),
            "apply_to_base_policy_variants": True,
        },
        "ict_short_continuation_xgb_v1": {
            "abstain_session_regimes": ("asia", "off_hours"),
            "abstain_composite_regimes": (
                "ranging_low",
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "strong_up_medium",
            ),
            "apply_to_base_policy_variants": True,
        },
        "ict_short_meta_xgb_v1": {
            "apply_to_base_policy_variants": True,
        },
        "ict_short_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "ranging_medium",
                "strong_up_low",
            ),
            "apply_to_base_policy_variants": True,
        },
    },
    "ict_es_primary_prune_v2": {
        "ict_long_reversal_xgb_v1": {
            "abstain_composite_regimes": ("ranging_low",),
            "apply_to_base_policy_variants": True,
        },
        "ict_short_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "ranging_medium",
                "strong_up_low",
            ),
            "apply_to_base_policy_variants": True,
        },
    },
    "ict_es_primary_prune_v3": {
        "ict_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "strong_down_low",
                "strong_up_low",
            ),
            "apply_to_base_policy_variants": True,
        },
        "ict_short_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "ranging_medium",
                "strong_up_low",
            ),
            "apply_to_base_policy_variants": True,
        },
    },
    "ict_es_primary_prune_v4_long_reversal_surgical": {
        "ict_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "strong_down_low",
                "strong_up_low",
            ),
            "abstain_composite_session_pairs": (("ranging_high", "new_york"),),
            "apply_to_base_policy_variants": True,
        },
        "ict_short_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_low",
                "ranging_medium",
                "strong_up_low",
            ),
            "apply_to_base_policy_variants": True,
        },
    },
    "long_breakout_regime_prune_v1": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
        }
    },
    "long_breakout_regime_prune_v2": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
                "strong_up_medium",
            ),
        }
    },
    "long_breakout_regime_prune_v1_q20": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "minimum_probability_quantile": 0.20,
        }
    },
    "long_breakout_regime_prune_v1_q20_asia": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "abstain_session_regimes": ("asia",),
            "minimum_probability_quantile": 0.20,
        }
    },
    "long_breakout_regime_prune_v1_q25": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "minimum_probability_quantile": 0.25,
        }
    },
    "long_breakout_regime_prune_v1_q30": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "minimum_probability_quantile": 0.30,
        }
    },
    "long_breakout_regime_prune_v3": {
        "long_breakout_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_high",
                "ranging_low",
                "strong_up_low",
            ),
            "abstain_composite_session_pairs": (("strong_up_medium", "asia"),),
            "abstain_composite_stress_pairs": (("strong_up_medium", "elevated"),),
            "minimum_probability_quantile": 0.20,
        }
    },
    "short_meta_regime_prune_v1": {
        "short_ote_meta_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_medium",
            ),
        }
    },
    "short_meta_regime_prune_q20_v1": {
        "short_ote_meta_tcn_champion": {
            "abstain_composite_regimes": (
                "strong_down_high",
                "strong_down_low",
                "strong_down_medium",
                "ranging_medium",
            ),
            "minimum_probability_quantile": 0.20,
        }
    },
    "short_reversal_xgb_ranging_medium_london_prune_v1": {
        "short_reversal_xgb_v1": {
            "abstain_composite_session_pairs": (("ranging_medium", "london"),),
        }
    },
    "short_reversal_xgb_ranging_medium_london_hard_prune_v1": {
        "short_reversal_xgb_v1": {
            "abstain_composite_session_pairs": (("ranging_medium", "london"),),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_continuation_xgb_london_drawdown_prune_v1": {
        "frvp_long_continuation_xgb_v1": {
            "abstain_composite_session_pairs": (
                ("strong_up_medium", "london"),
                ("strong_up_high", "london"),
                ("strong_down_medium", "london"),
                ("ranging_low", "london"),
                ("ranging_medium", "london"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_continuation_xgb_overlap_composite_prune_v2": {
        "frvp_long_continuation_xgb_v1": {
            "abstain_session_regimes": ("overlap",),
            "abstain_composite_regimes": (
                "strong_down_medium",
                "strong_up_high",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_medium", "london"),
                ("strong_up_high", "london"),
                ("strong_down_medium", "london"),
                ("ranging_low", "london"),
                ("ranging_medium", "london"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_continuation_xgb_overlap_composite_prune_v3": {
        "frvp_long_continuation_xgb_v1": {
            "abstain_session_regimes": ("overlap",),
            "abstain_composite_regimes": (
                "strong_down_medium",
                "strong_up_high",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_medium", "london"),
                ("strong_up_high", "london"),
                ("strong_down_medium", "london"),
                ("ranging_low", "london"),
                ("ranging_medium", "london"),
                ("ranging_medium", "new_york"),
                ("strong_up_low", "asia"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_meta_xgb_composite_prune_v1": {
        "frvp_long_meta_xgb_v1": {
            "abstain_composite_regimes": (
                "strong_down_medium",
                "strong_up_medium",
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_meta_xgb_composite_prune_v2": {
        "frvp_long_meta_xgb_v1": {
            "abstain_composite_regimes": (
                "strong_down_medium",
                "strong_up_medium",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_high", "overlap"),
                ("strong_down_high", "new_york"),
                ("ranging_high", "new_york"),
                ("strong_up_low", "asia"),
                ("strong_up_low", "london"),
                ("ranging_low", "london"),
                ("ranging_medium", "new_york"),
                ("ranging_medium", "asia"),
                ("strong_down_low", "london"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_meta_xgb_composite_prune_v3": {
        "frvp_long_meta_xgb_v1": {
            "abstain_composite_regimes": (
                "strong_down_medium",
                "strong_up_medium",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_high", "overlap"),
                ("strong_down_high", "new_york"),
                ("ranging_high", "new_york"),
                ("strong_up_low", "asia"),
                ("strong_up_low", "london"),
                ("ranging_low", "london"),
                ("ranging_medium", "new_york"),
                ("ranging_medium", "asia"),
                ("strong_down_low", "london"),
                ("ranging_medium", "london"),
                ("strong_up_high", "london"),
                ("strong_down_high", "overlap"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_reversal_xgb_composite_prune_v1": {
        "frvp_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_high",
                "strong_up_low",
                "strong_up_medium",
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_reversal_xgb_composite_prune_v2": {
        "frvp_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_high",
                "strong_up_low",
                "strong_up_medium",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_high", "new_york"),
                ("strong_down_high", "new_york"),
                ("ranging_low", "london"),
                ("strong_down_low", "asia"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_reversal_xgb_composite_prune_v3": {
        "frvp_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_high",
                "strong_up_low",
                "strong_up_medium",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_high", "new_york"),
                ("strong_down_high", "new_york"),
                ("ranging_low", "london"),
                ("strong_down_low", "asia"),
                ("ranging_medium", "asia"),
                ("ranging_medium", "london"),
                ("strong_down_low", "london"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_reversal_xgb_recent_regime_prune_v1": {
        "frvp_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_high",
                "strong_up_low",
                "strong_up_medium",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_high", "new_york"),
                ("strong_down_high", "new_york"),
                ("ranging_low", "london"),
                ("strong_down_low", "asia"),
                ("ranging_medium", "asia"),
                ("ranging_medium", "london"),
                ("strong_down_low", "london"),
                ("strong_down_medium", "asia"),
                ("ranging_low", "asia"),
                ("ranging_medium", "new_york"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
    "frvp_long_reversal_xgb_recent_regime_prune_v2": {
        "frvp_long_reversal_xgb_v1": {
            "abstain_composite_regimes": (
                "ranging_high",
                "strong_up_low",
                "strong_up_medium",
            ),
            "abstain_composite_session_pairs": (
                ("strong_up_high", "new_york"),
                ("strong_down_high", "new_york"),
                ("ranging_low", "london"),
                ("strong_down_low", "asia"),
                ("ranging_medium", "asia"),
                ("ranging_medium", "london"),
                ("strong_down_low", "london"),
                ("strong_down_medium", "asia"),
                ("ranging_low", "asia"),
                ("ranging_medium", "new_york"),
                ("strong_down_high", "overlap"),
            ),
            "apply_to_base_policy_variants": True,
        }
    },
}


def resolve_targeted_filters(
    model_id: str,
    preset_name: str | None,
) -> Dict[str, object]:
    if preset_name is None:
        return {}
    preset = TARGETED_FILTER_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown targeted_filter_preset: {preset_name!r}")
    return dict(preset.get(model_id, {}))


def build_targeted_abstain_config(
    model: OTEModelRecord,
    threshold_config: ThresholdSearchConfig,
    *,
    targeted_filter_preset: str | None = None,
) -> HardAbstainConfig:
    metadata = model.abstain_policy or {}
    targeted_filters = resolve_targeted_filters(model.model_id, targeted_filter_preset)
    session_spread_pips = _resolve_session_spread_units(metadata, threshold_config)
    expected_move_by_regime = metadata.get("expected_move_by_regime")
    return HardAbstainConfig(
        abstain_high_stress=bool(metadata.get("abstain_high_stress", True)),
        abstain_off_hours=bool(metadata.get("abstain_off_hours", True)),
        cooldown_bars=int(metadata.get("cooldown_bars", threshold_config.event_cooldown_bars)),
        minimum_expected_move_to_spread=float(metadata.get("minimum_expected_move_to_spread", 2.0)),
        session_spread_pips=session_spread_pips,
        expected_move_by_regime=(
            {str(key): float(value) for key, value in dict(expected_move_by_regime).items()}
            if isinstance(expected_move_by_regime, dict)
            else None
        ),
        abstain_session_regimes=tuple(
            str(value)
            for value in (
                targeted_filters.get("abstain_session_regimes")
                or metadata.get("abstain_session_regimes")
                or ()
            )
        ),
        abstain_composite_regimes=tuple(
            str(value)
            for value in (
                targeted_filters.get("abstain_composite_regimes")
                or metadata.get("abstain_composite_regimes")
                or ()
            )
        ),
        abstain_composite_session_pairs=_coerce_pair_filters(
            targeted_filters.get("abstain_composite_session_pairs")
            or metadata.get("abstain_composite_session_pairs")
            or ()
        ),
        abstain_composite_stress_pairs=_coerce_pair_filters(
            targeted_filters.get("abstain_composite_stress_pairs")
            or metadata.get("abstain_composite_stress_pairs")
            or ()
        ),
        minimum_probability_quantile=(
            float(targeted_filters["minimum_probability_quantile"])
            if "minimum_probability_quantile" in targeted_filters
            else (
                float(metadata["minimum_probability_quantile"])
                if "minimum_probability_quantile" in metadata and metadata["minimum_probability_quantile"] is not None
                else None
            )
        ),
        apply_to_base_policy_variants=bool(
            targeted_filters.get(
                "apply_to_base_policy_variants",
                metadata.get("apply_to_base_policy_variants", False),
            )
        ),
        probability_column="policy_probability",
        signal_candidate_column="policy_signal_candidate",
        position_column=threshold_config.position_column,
    )


def describe_abstain_config(config: HardAbstainConfig) -> Dict[str, object]:
    return {
        "abstain_high_stress": bool(config.abstain_high_stress),
        "abstain_off_hours": bool(config.abstain_off_hours),
        "cooldown_bars": int(config.cooldown_bars),
        "minimum_expected_move_to_spread": float(config.minimum_expected_move_to_spread),
        "abstain_session_regimes": [str(value) for value in config.abstain_session_regimes],
        "abstain_composite_regimes": [str(value) for value in config.abstain_composite_regimes],
        "abstain_composite_session_pairs": [
            [str(composite_regime), str(session_regime)]
            for composite_regime, session_regime in config.abstain_composite_session_pairs
        ],
        "abstain_composite_stress_pairs": [
            [str(composite_regime), str(stress_regime)]
            for composite_regime, stress_regime in config.abstain_composite_stress_pairs
        ],
        "minimum_probability_quantile": None
        if config.minimum_probability_quantile is None
        else float(config.minimum_probability_quantile),
        "apply_to_base_policy_variants": bool(config.apply_to_base_policy_variants),
    }


def _coerce_pair_filters(values: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, (list, tuple)):
        return ()

    normalized_pairs: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        normalized_pairs.append((str(value[0]), str(value[1])))
    return tuple(normalized_pairs)


def _resolve_session_spread_units(
    metadata: Dict[str, object],
    threshold_config: ThresholdSearchConfig,
) -> dict[str, float]:
    session_spread_pips = metadata.get("session_spread_pips")
    if not isinstance(session_spread_pips, dict):
        return {
            str(key): float(value)
            for key, value in dict(threshold_config.session_spread_pips).items()
        }

    metadata_instrument = str(metadata.get("instrument", "")).strip().lower()
    metadata_unit_label = str(metadata.get("unit_label", "")).strip().lower()
    threshold_instrument = str(threshold_config.instrument).strip().lower()
    threshold_unit_label = str(threshold_config.unit_label).strip().lower()

    is_legacy_fx_metadata = not metadata_instrument and not metadata_unit_label and threshold_instrument == "fx"
    is_matching_metadata = (
        metadata_instrument == threshold_instrument
        and metadata_unit_label == threshold_unit_label
    )

    if not is_legacy_fx_metadata and not is_matching_metadata:
        return {
            str(key): float(value)
            for key, value in dict(threshold_config.session_spread_pips).items()
        }

    return {
        str(key): float(value)
        for key, value in dict(session_spread_pips).items()
    }
