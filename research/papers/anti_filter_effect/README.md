# When Abstention Destroys Alpha

Research paper, audit memo, evidence map, and reproducible analysis for the Anti-Filter Effect, Counterfactual
Abstain Attribution (CAA), and the Signal Promotion Contract, grounded in this repository's saved FRVP, ICT, and OTE
artifacts.

```
research/papers/anti_filter_effect/
    README.md              this file
    audit.md               research audit memo (sections A-E) written before the paper
    evidence_map.csv       one row per empirical claim with source path and confidence class
    paper.tex              arXiv-style paper (single-column article)
    references.bib         verified references only
    paper.pdf              built output (see step 3)
    scripts/
        analyze_attribution.py   CAA over saved trade tapes and policy evaluations
        build_paper_assets.py    figures (PDF) and LaTeX table fragments from the CAA outputs
    artifacts/
        attribution_summary.csv         accepted/rejected/baseline statistics per tape x abstention rule
        attribution_breakdowns.csv      accepted vs rejected by side, session, regime, year, fold, label, outcome
        trade_membership.csv            per-trade accepted/rejected membership and cutoff (selected rules)
        top_contributors.csv            top-5 net winners in each population
        walk_forward_pair_census.csv    nested base-vs-abstain pairs, per-fold test split, all saved walk-forward roots
        static_pair_census.csv          nested pairs on static held-out test splits
        curated_run_pairs.csv           baseline-vs-filtered saved runs matched on (entry_datetime, source_row_idx)
        parity_evidence.json            replay/parity numbers copied from the paper-signal validation summaries
        source_integrity_checks.csv     saved summary totals vs recomputed tape totals
        analysis_checks.json            census counts and identity checks
        attribution_source_manifest.csv SHA-256 of every source artifact read
        figures/*.pdf, tables/*.tex
```

Nothing under `artifacts/` is a canonical model, label, policy, or execution artifact. The scripts only read
`model_testing/reports/**` and `ote_live/runtime_manifests/**` and write here.

## Reproduction (PowerShell, from the repository root)

### 1. Attribution analysis

```powershell
ote_venv\Scripts\python.exe research\papers\anti_filter_effect\scripts\analyze_attribution.py
```

Runtime about 30 seconds. Prints `analysis_checks.json`. Asserts that every tape reproduces its saved
`summary.json` trade count and net PnL, that accepted + rejected partitions reconcile exactly, and that the 24
retrospective-quantile rows match the 2026-08-30 historical screen.

### 2. Figures and tables

```powershell
ote_venv\Scripts\python.exe research\papers\anti_filter_effect\scripts\build_paper_assets.py
```

Writes `artifacts\figures\{coverage_profit_curves,policy_pair_scatter,confidence_quintiles}.pdf` and
`artifacts\tables\{headline_attribution,census,cutoff_sensitivity,parity}.tex`.

### 3. LaTeX build

MiKTeX (user-scope) was installed for this build with `winget install --id MiKTeX.MiKTeX -e --scope user`.
If `pdflatex` is on `PATH` the prefix can be dropped.

```powershell
$bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
Push-Location research\papers\anti_filter_effect
& "$bin\pdflatex.exe" -interaction=nonstopmode paper.tex
& "$bin\bibtex.exe" paper
& "$bin\pdflatex.exe" -interaction=nonstopmode paper.tex
& "$bin\pdflatex.exe" -interaction=nonstopmode paper.tex
Pop-Location
```

### 4. Primary evidence locations

| Evidence | Path |
| --- | --- |
| OTE hard-abstain pairs, static test | `model_testing\reports\ote_threshold_policies\v1_v2_tcn_focus\policy_evaluation.csv` |
| OTE hard-abstain pairs, 49-fold walk-forward | `model_testing\reports\ote_policy_backtests\v1_v2_tcn_focus\<model>\policy_evaluation.csv` |
| OTE live-registry pairs | `model_testing\reports\ote_policy_backtests\multifamily_live_v1\<model>\policy_evaluation.csv` |
| FRVP refresh tapes and pairs | `model_testing\reports\frvp_backtests\frvp_es_primary_refresh_20260701\<model>\` |
| ICT leakage-safe tapes and pairs | `model_testing\reports\ict_backtests\ict_es_primary_bootstrap_20260726_full\<model>\` |
| Probability-quantile study (2026-08-30) | `model_testing\reports\probability_quantile_sweeps\frvp_ict_weak_discrimination_20260830\` and `model_testing\reports\ict_backtests\ict_probability_quantile_candidate_sweep_20260830\` |
| FRVP regime/session prunes | `model_testing\reports\frvp_backtests\frvp_long_*_gatefix_*` and `model_testing\reports\frvp_regime_gated_deployment\frvp_regime_gated_deployment_20260721\` |
| Fixed-live-policy sensitivity and replay parity | `model_testing\reports\frvp_paper_signal_bundles\frvp_es_paper_signal_20260816\run_summary.json`, `ote_live\runtime_manifests\frvp_es_paper_signal_20260816\paper_signal_validation_summary.json`, `ote_live\runtime_manifests\ict_es_paper_signal_20260813\paper_signal_validation_summary.json` |
| Breakout-sustained pilot (AP 0.98, negative PnL) | `models\experiment_a_breakout_sustained_pilot_20260621\long_breakout_sustained\training_summary.json`, `model_testing\reports\ote_policy_backtests\experiment_a_breakout_sustained_pilot_20260621\model_summary.csv` |
| Placebo readout | `model_testing\reports\frvp_placebo_readouts\frvp_long_continuation_xgb_v1_20260717\placebo_readout_summary.json` |
| Leakage audit | `model_testing\reports\ict_leakage_control_audits\ict_es_primary_refresh_20260724_spacing_refit_final_confirm_leakage_audit_after_embargo\ict_leakage_control_summary.md` |
| Journals | `docs\FRVP_experiment_journal.md`, `docs\ICT_experiment_journal.md`, `docs\FRVP_ICT_probability_quantile_filter_study_20260830.md`, `docs\POLICY_LAYER_AND_ARCHITECTURE_RESEARCH.md` |
