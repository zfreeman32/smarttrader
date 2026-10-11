# Documentation index

Consolidated on 2026-10-09: about 75 documents were merged, verbatim, into the topic documents below. Each merged
source keeps an HTML anchor named after its old file stem, so `OLD_NAME.md` content is at `NEW.md#old_name`. The table
at the end maps every old path. Byte-exact originals are on the local git branch `backup/pre-cleanup-20261009`.

## Active research program (2026-09 onward)

| Document | Contents |
|---|---|
| [section4_progress.md](section4_progress.md) | **Current state.** Section 4 live-parity work, the feature-contract v2 retrain (R0-R3), the B3 threshold sweep and the 2026-10-08 B1-outcome label research; with the parity results, training-side recommendation and R1 design inline. |
| [B1_outcome_contracts.md](B1_outcome_contracts.md) | B1 outcome contracts: design, audit, FRVP risk, scope/ATR, OTE evaluation, completed matrix, validation, candidate review. |
| [B2_model_repair.md](B2_model_repair.md) | B2 lineage, data repair, retraining plan and reviews, S3 backtests, ES expected-move contract, candidate source-history repair. |
| [B3_B4_freeze_and_routes.md](B3_B4_freeze_and_routes.md) | B3 comparison protocols, B4 freezes and route bindings (v1/v2). |
| [section3_simulation_rules.md](section3_simulation_rules.md) | Section 3 execution rules, gap matrix and smoke replay. |
| [B5_B6_live_feature_parity_20260929.md](B5_B6_live_feature_parity_20260929.md), [section4_inventory_20261002.md](section4_inventory_20261002.md), [section4_parity_protocol_v1_draft_20261002.md](section4_parity_protocol_v1_draft_20261002.md) | Kept byte-identical: the locked section 4 parity protocol pins them by SHA-256. |
| [live_app_change_journal.md](live_app_change_journal.md) | Every live-app change (A1-A11 and later). The IBKR ES feed setup now lives in `ote_live/README.md`. |

## Strategy journals and design papers

| Document | Contents |
|---|---|
| [FRVP_experiment_journal.md](FRVP_experiment_journal.md) | Canonical FRVP experiment record; includes the record template, operational contract, regime-gated roadmap, setup-specific rerun plan and leaderboard. |
| [FRVP_ES_primary_audit_report.md](FRVP_ES_primary_audit_report.md) | FRVP ES diagnostic audit report. |
| [FRVP_ES_primary_6E_variant_design.md](FRVP_ES_primary_6E_variant_design.md) | FRVP ES design paper, with the original EURUSD FRVP paper as an appendix. |
| [ICT_execution_journal.md](ICT_execution_journal.md) | Canonical ICT record; the former ICT experiment journal (E1-E3 and dated entries) is interleaved by date. |
| [ICT_ES_META_LABELING_DESIGN.md](ICT_ES_META_LABELING_DESIGN.md) | ICT meta-labeling design paper. |
| [FRVP_ICT_reference_and_audits.md](FRVP_ICT_reference_and_audits.md) | FRVP/ICT/OTE trading logic, concept-to-code audit, live signal-output audit, probability quantile filter study. |
| [OTE_journal.md](OTE_journal.md) | EURUSD OTE history: original architecture vision, runbook, production plan, 8-model plan, retrain plans, 1-minute plan, policy-layer research. |
| [model_backtest_charts.md](model_backtest_charts.md) | Backtest atlas, realized P&L charts and the 2026-05/07 model leaderboards. |

## Other folders

- `audits/`: machine-readable evidence (JSON/CSV) cited by the documents and frozen configs.
- `audits/data_prune_20261011.csv`: every gitignored data path removed on 2026-10-11 (133 GB: superseded B2-v1
  data, old experiment model bundles and registries, bulk report CSVs, derived datasets). Evidence records
  that cite those paths remain as documentation, but the bytes are gone and cannot be re-hashed.
- `assets/`: charts and roster tables used by the documents.
- `papers/`: external reference papers and the EURUSD OTE framework paper drafts.
- The Anti-Filter Effect paper lives in `research/papers/anti_filter_effect/`.

## Where old documents went

| Old path | New location |
|---|---|
| `docs/B1_baseline_scope_and_model_setup_audit_20260917.md` | [docs/B1_outcome_contracts.md#b1_baseline_scope_and_model_setup_audit_20260917](B1_outcome_contracts.md#b1_baseline_scope_and_model_setup_audit_20260917) |
| `docs/B1_candidate_outcome_review_20260929.md` | [docs/B1_outcome_contracts.md#b1_candidate_outcome_review_20260929](B1_outcome_contracts.md#b1_candidate_outcome_review_20260929) |
| `docs/B1_completion_validation_20260918.md` | [docs/B1_outcome_contracts.md#b1_completion_validation_20260918](B1_outcome_contracts.md#b1_completion_validation_20260918) |
| `docs/B1_FRVP_risk_resolution_20260917.md` | [docs/B1_outcome_contracts.md#b1_frvp_risk_resolution_20260917](B1_outcome_contracts.md#b1_frvp_risk_resolution_20260917) |
| `docs/B1_OTE_evaluation_resolution_20260918.md` | [docs/B1_outcome_contracts.md#b1_ote_evaluation_resolution_20260918](B1_outcome_contracts.md#b1_ote_evaluation_resolution_20260918) |
| `docs/B1_outcome_contract_audit_20260916.md` | [docs/B1_outcome_contracts.md#b1_outcome_contract_audit_20260916](B1_outcome_contracts.md#b1_outcome_contract_audit_20260916) |
| `docs/B1_outcome_contract_design_20260916.md` | [docs/B1_outcome_contracts.md#b1_outcome_contract_design_20260916](B1_outcome_contracts.md#b1_outcome_contract_design_20260916) |
| `docs/B1_outcome_contract_matrix_20260916.md` | [docs/B1_outcome_contracts.md#b1_outcome_contract_matrix_20260916](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260916) |
| `docs/B1_outcome_contract_matrix_20260917.md` | [docs/B1_outcome_contracts.md#b1_outcome_contract_matrix_20260917](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260917) |
| `docs/B1_outcome_contract_matrix_20260918.md` | [docs/B1_outcome_contracts.md#b1_outcome_contract_matrix_20260918](B1_outcome_contracts.md#b1_outcome_contract_matrix_20260918) |
| `docs/B1_outcome_contract_matrix_complete_20260918.md` | [docs/B1_outcome_contracts.md#b1_outcome_contract_matrix_complete_20260918](B1_outcome_contracts.md#b1_outcome_contract_matrix_complete_20260918) |
| `docs/B2_candidate_history_audit_20260928.md` | [docs/B2_model_repair.md#b2_candidate_history_audit_20260928](B2_model_repair.md#b2_candidate_history_audit_20260928) |
| `docs/B2_candidate_history_catchup_5_14_18_23_20261001.md` | [docs/B2_model_repair.md#b2_candidate_history_catchup_5_14_18_23_20261001](B2_model_repair.md#b2_candidate_history_catchup_5_14_18_23_20261001) |
| `docs/B2_candidate_history_minimum_and_expanded_core_20260930.md` | [docs/B2_model_repair.md#b2_candidate_history_minimum_and_expanded_core_20260930](B2_model_repair.md#b2_candidate_history_minimum_and_expanded_core_20260930) |
| `docs/B2_exploratory_review_20260924.md` | [docs/B2_model_repair.md#b2_exploratory_review_20260924](B2_model_repair.md#b2_exploratory_review_20260924) |
| `docs/B2_first_pass_review_and_queue_20260922.md` | [docs/B2_model_repair.md#b2_first_pass_review_and_queue_20260922](B2_model_repair.md#b2_first_pass_review_and_queue_20260922) |
| `docs/B2_lineage_inventory_and_dependency_trace_20260918.md` | [docs/B2_model_repair.md#b2_lineage_inventory_and_dependency_trace_20260918](B2_model_repair.md#b2_lineage_inventory_and_dependency_trace_20260918) |
| `docs/B2_minimum_repairs_20260919.md` | [docs/B2_model_repair.md#b2_minimum_repairs_20260919](B2_model_repair.md#b2_minimum_repairs_20260919) |
| `docs/B2_model_retention_roster_20260920.md` | [docs/B2_model_repair.md#b2_model_retention_roster_20260920](B2_model_repair.md#b2_model_retention_roster_20260920) |
| `docs/B2_model_retraining_plan_20260922.md` | [docs/B2_model_repair.md#b2_model_retraining_plan_20260922](B2_model_repair.md#b2_model_retraining_plan_20260922) |
| `docs/B2_research_scope_20260929.md` | [docs/B2_model_repair.md#b2_research_scope_20260929](B2_model_repair.md#b2_research_scope_20260929) |
| `docs/B2_retraining_readiness_audit_20260922.md` | [docs/B2_model_repair.md#b2_retraining_readiness_audit_20260922](B2_model_repair.md#b2_retraining_readiness_audit_20260922) |
| `docs/B2_S3_backtest_20260927.md` | [docs/B2_model_repair.md#b2_s3_backtest_20260927](B2_model_repair.md#b2_s3_backtest_20260927) |
| `docs/B2_S3_corrected_backtest_20260927.md` | [docs/B2_model_repair.md#b2_s3_corrected_backtest_20260927](B2_model_repair.md#b2_s3_corrected_backtest_20260927) |
| `docs/B2_S3_full_policy_backtest_20260928.md` | [docs/B2_model_repair.md#b2_s3_full_policy_backtest_20260928](B2_model_repair.md#b2_s3_full_policy_backtest_20260928) |
| `docs/B2_S3_source_history_repair_20260927.md` | [docs/B2_model_repair.md#b2_s3_source_history_repair_20260927](B2_model_repair.md#b2_s3_source_history_repair_20260927) |
| `docs/B2_shared_data_repair_20260921.md` | [docs/B2_model_repair.md#b2_shared_data_repair_20260921](B2_model_repair.md#b2_shared_data_repair_20260921) |
| `docs/B2_validation_repair_20260922.md` | [docs/B2_model_repair.md#b2_validation_repair_20260922](B2_model_repair.md#b2_validation_repair_20260922) |
| `docs/B3_prospective_comparison_20260929.md` | [docs/B3_B4_freeze_and_routes.md#b3_prospective_comparison_20260929](B3_B4_freeze_and_routes.md#b3_prospective_comparison_20260929) |
| `docs/B3_prospective_comparison_v2_20261002.md` | [docs/B3_B4_freeze_and_routes.md#b3_prospective_comparison_v2_20261002](B3_B4_freeze_and_routes.md#b3_prospective_comparison_v2_20261002) |
| `docs/B4_baseline_freeze_20260929.md` | [docs/B3_B4_freeze_and_routes.md#b4_baseline_freeze_20260929](B3_B4_freeze_and_routes.md#b4_baseline_freeze_20260929) |
| `docs/B4_broad_full_session_shadow_freeze_v1_20261001.md` | [docs/B3_B4_freeze_and_routes.md#b4_broad_full_session_shadow_freeze_v1_20261001](B3_B4_freeze_and_routes.md#b4_broad_full_session_shadow_freeze_v1_20261001) |
| `docs/B4_broad_full_session_shadow_freeze_v2_20261001.md` | [docs/B3_B4_freeze_and_routes.md#b4_broad_full_session_shadow_freeze_v2_20261001](B3_B4_freeze_and_routes.md#b4_broad_full_session_shadow_freeze_v2_20261001) |
| `docs/broad_full_session_shadow_route_bindings_v1_20261001.md` | [docs/B3_B4_freeze_and_routes.md#broad_full_session_shadow_route_bindings_v1_20261001](B3_B4_freeze_and_routes.md#broad_full_session_shadow_route_bindings_v1_20261001) |
| `docs/broad_full_session_shadow_route_bindings_v2_20261001.md` | [docs/B3_B4_freeze_and_routes.md#broad_full_session_shadow_route_bindings_v2_20261001](B3_B4_freeze_and_routes.md#broad_full_session_shadow_route_bindings_v2_20261001) |
| `docs/catchup_ids_5_23_14_18_progress.md` | [docs/B2_model_repair.md#catchup_ids_5_23_14_18_progress](B2_model_repair.md#catchup_ids_5_23_14_18_progress) |
| `docs/end-to-end-README.md` | [docs/OTE_journal.md#end-to-end-readme](OTE_journal.md#end-to-end-readme) |
| `docs/ES_expected_move_contract_20260927.md` | [docs/B2_model_repair.md#es_expected_move_contract_20260927](B2_model_repair.md#es_expected_move_contract_20260927) |
| `docs/EURUSD_OTE_Framework.docx` | [docs/papers/EURUSD_OTE_Framework.docx](papers/EURUSD_OTE_Framework.docx) |
| `docs/EURUSD_OTE_Framework.pdf` | removed: earlier render of `papers/EURUSD_OTE_Framework.docx`, superseded by `papers/EURUSD_OTE_FRAMEWORK_paper_v2.pdf` |
| `docs/EURUSD_OTE_FRAMEWORK_paper_v2.pdf` | [docs/papers/EURUSD_OTE_FRAMEWORK_paper_v2.pdf](papers/EURUSD_OTE_FRAMEWORK_paper_v2.pdf) |
| `docs/FRVP_experiment_record_template.md` | [docs/FRVP_experiment_journal.md#frvp_experiment_record_template](FRVP_experiment_journal.md#frvp_experiment_record_template) |
| `docs/FRVP_ICT_CONCEPT_TO_CODE_AUDIT.md` | [docs/FRVP_ICT_reference_and_audits.md#frvp_ict_concept_to_code_audit](FRVP_ICT_reference_and_audits.md#frvp_ict_concept_to_code_audit) |
| `docs/FRVP_ICT_Live_Signal_Output_Audit_20260811.md` | [docs/FRVP_ICT_reference_and_audits.md#frvp_ict_live_signal_output_audit_20260811](FRVP_ICT_reference_and_audits.md#frvp_ict_live_signal_output_audit_20260811) |
| `docs/FRVP_ICT_OTE_TRADING_LOGIC.md` | [docs/FRVP_ICT_reference_and_audits.md#frvp_ict_ote_trading_logic](FRVP_ICT_reference_and_audits.md#frvp_ict_ote_trading_logic) |
| `docs/FRVP_ICT_probability_quantile_filter_study_20260830.md` | [docs/FRVP_ICT_reference_and_audits.md#frvp_ict_probability_quantile_filter_study_20260830](FRVP_ICT_reference_and_audits.md#frvp_ict_probability_quantile_filter_study_20260830) |
| `docs/FRVP_ML_Strategy.md` | [docs/FRVP_ES_primary_6E_variant_design.md#frvp_ml_strategy](FRVP_ES_primary_6E_variant_design.md#frvp_ml_strategy) |
| `docs/FRVP_operational_contract_20260721.md` | [docs/FRVP_experiment_journal.md#frvp_operational_contract_20260721](FRVP_experiment_journal.md#frvp_operational_contract_20260721) |
| `docs/FRVP_regime_gated_deployment_roadmap_20260721.md` | [docs/FRVP_experiment_journal.md#frvp_regime_gated_deployment_roadmap_20260721](FRVP_experiment_journal.md#frvp_regime_gated_deployment_roadmap_20260721) |
| `docs/FRVP_setup_specific_long_rerun_plan_20260829.md` | [docs/FRVP_experiment_journal.md#frvp_setup_specific_long_rerun_plan_20260829](FRVP_experiment_journal.md#frvp_setup_specific_long_rerun_plan_20260829) |
| `docs/FRVP_setup_specific_model_leaderboard_20260830.md` | [docs/FRVP_experiment_journal.md#frvp_setup_specific_model_leaderboard_20260830](FRVP_experiment_journal.md#frvp_setup_specific_model_leaderboard_20260830) |
| `model_training/ote_training/FULL_PIPELINE_README.md` | [model_training/ote_training/README.md#full_pipeline_readme](../model_training/ote_training/README.md#full_pipeline_readme) |
| `docs/IBKR_ES_live_data_setup.md` | [ote_live/README.md#ibkr_es_live_data_setup](../ote_live/README.md#ibkr_es_live_data_setup) |
| `docs/ICT_experiment_journal.md` | [docs/ICT_execution_journal.md#ict_experiment_journal](ICT_execution_journal.md#ict_experiment_journal) |
| `models/LIVE_MODEL_CATALOG.md` | [models/README.md#live_model_catalog](../models/README.md#live_model_catalog) |
| `docs/model_backtest_atlas_20260927.md` | [docs/model_backtest_charts.md#model_backtest_atlas_20260927](model_backtest_charts.md#model_backtest_atlas_20260927) |
| `model_testing/reports/MODEL_LEADERBOARDS_20260527.md` | [docs/model_backtest_charts.md#model_leaderboards_20260527](model_backtest_charts.md#model_leaderboards_20260527) |
| `model_testing/reports/MODEL_LEADERBOARDS_20260730.md` | [docs/model_backtest_charts.md#model_leaderboards_20260730](model_backtest_charts.md#model_leaderboards_20260730) |
| `docs/MODEL_PROMOTION_AND_OTE_LIVE_8_MODEL_PLAN.md` | [docs/OTE_journal.md#model_promotion_and_ote_live_8_model_plan](OTE_journal.md#model_promotion_and_ote_live_8_model_plan) |
| `docs/model_realized_pnl_charts_20260927.md` | [docs/model_backtest_charts.md#model_realized_pnl_charts_20260927](model_backtest_charts.md#model_realized_pnl_charts_20260927) |
| `docs/Model_Retrain_Plan_1min.md` | [docs/OTE_journal.md#model_retrain_plan_1min](OTE_journal.md#model_retrain_plan_1min) |
| `docs/Optimal_Trade_Entry_Detection_Pipeline.docx.txt` | [docs/papers/Optimal_Trade_Entry_Detection_Pipeline.docx.txt](papers/Optimal_Trade_Entry_Detection_Pipeline.docx.txt) |
| `model_testing/OTE_MULTI_FAMILY_TCN_V1_V2_RETRAIN_PLAN.md` | [docs/OTE_journal.md#ote_multi_family_tcn_v1_v2_retrain_plan](OTE_journal.md#ote_multi_family_tcn_v1_v2_retrain_plan) |
| `model_training/ote_training/OTE_TRAINING_WORKFLOW_REPORT.md` | [docs/OTE_journal.md#ote_training_workflow_report](OTE_journal.md#ote_training_workflow_report) |
| `docs/POLICY_LAYER_AND_ARCHITECTURE_RESEARCH.md` | [docs/OTE_journal.md#policy_layer_and_architecture_research](OTE_journal.md#policy_layer_and_architecture_research) |
| `model_testing/POST_TRAINING_PRODUCTION_PLAN.md` | [docs/OTE_journal.md#post_training_production_plan](OTE_journal.md#post_training_production_plan) |
| `docs/predictableequitymarkets.pdf` | [docs/papers/predictableequitymarkets.pdf](papers/predictableequitymarkets.pdf) |
| `docs/PRODUCTION_REALTIME_SIGNAL_SYSTEM_PLAN.md` | [docs/OTE_journal.md#production_realtime_signal_system_plan](OTE_journal.md#production_realtime_signal_system_plan) |
| `docs/Quantitative_Trading_ML_Pipeline.docx.txt` | [docs/papers/Quantitative_Trading_ML_Pipeline.docx.txt](papers/Quantitative_Trading_ML_Pipeline.docx.txt) |
| `docs/research_paper.docx` | [docs/papers/research_paper.docx](papers/research_paper.docx) |
| `docs/retrain_r1_design_20261005.md` | [docs/section4_progress.md#retrain_r1_design_20261005](section4_progress.md#retrain_r1_design_20261005) |
| `docs/section3_gap_matrix_20261002.md` | [docs/section3_simulation_rules.md#section3_gap_matrix_20261002](section3_simulation_rules.md#section3_gap_matrix_20261002) |
| `docs/section3_progress.md` | [docs/section3_simulation_rules.md#section3_progress](section3_simulation_rules.md#section3_progress) |
| `docs/section3_smoke_replay_20261002.md` | [docs/section3_simulation_rules.md#section3_smoke_replay_20261002](section3_simulation_rules.md#section3_smoke_replay_20261002) |
| `docs/section3_smoke_replay_20261002_v2.md` | [docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v2](section3_simulation_rules.md#section3_smoke_replay_20261002_v2) |
| `docs/section3_smoke_replay_20261002_v3.md` | [docs/section3_simulation_rules.md#section3_smoke_replay_20261002_v3](section3_simulation_rules.md#section3_smoke_replay_20261002_v3) |
| `docs/section4_parity_20261005.md` | [docs/section4_progress.md#section4_parity_20261005](section4_progress.md#section4_parity_20261005) |
| `docs/section4_training_side_recommendation_20261005.md` | [docs/section4_progress.md#section4_training_side_recommendation_20261005](section4_progress.md#section4_training_side_recommendation_20261005) |
| `docs/synthetic-time-series-research.pdf` | [docs/papers/synthetic-time-series-research.pdf](papers/synthetic-time-series-research.pdf) |
| `docs/Volume profile indicators_ basic concepts — TradingView.pdf` | [docs/papers/Volume profile indicators_ basic concepts — TradingView.pdf](<papers/Volume profile indicators_ basic concepts — TradingView.pdf>) |
