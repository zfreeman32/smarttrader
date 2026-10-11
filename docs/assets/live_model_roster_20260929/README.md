# Model experiment database

Open `chart.html` directly in a browser. No server, CDN, or internet connection is needed.

- **In app:** exact training artifact paths referenced by configured runtime manifests. This is a configuration snapshot, not a running-process check. Archived/paused audit status is shown independently of app membership.
- **Potential:** the six repaired research candidates selected by the September 29 roster. This is a research classification, not a profitability or promotion claim.
- **All experiments:** discovered training bundles and all 264 indexed policy backtest runs. Search, filter, sort, export, and open a training record for its complete saved configuration, metrics, evidence gaps, and notes.
- **Leaderboard:** repaired-candidate test AUC, historical policy results separately in ticks or pips, and the corrected S3 USD result. Different policies and periods are not normalized for fair comparison.
- **P&L gallery:** all existing PNG charts from `realized_pnl` and `backtest_atlas`, plus interactive cumulative realized USD curves for the corrected S3 branches. These curves use closed-trade order, not elapsed time, on the horizontal axis.
- **Coverage & sources:** long/short setup coverage, experiment journals, interpretation notes, and input SHA-256 fingerprints.

The catalog inventories every `training_summary.json` under `models/` and `model_testing/reports/`, plus every bundle named by the retention roster, selected roster, or configured runtime manifests. Policy runs come from `backtest_curve_index.csv`; journal-only ideas remain available through the linked journals. “Not collected / verified” means the catalog cannot establish version-bound P&L; it does not claim that no historical model-ID result exists.

Research notes and priorities are saved in browser local storage, keyed by the artifact path. Use **Export research notes** to back them up and **Import notes** to merge a backup. They do not alter evidence or application settings. Moving the HTML file or using another browser may require importing the backup.

To regenerate the selected roster evidence and interactive database from the repository root:

```powershell
ote_venv/Scripts/python.exe scripts/build_live_model_roster_chart.py
```

To refresh the broad inventory and page using the existing selected-roster snapshot:

```powershell
python scripts/build_model_experiment_database.py
```

Edit the UI in `scripts/model_experiment_database.html`, then rebuild. `chart.html` and `catalog.json` are generated artifacts. New training bundles are picked up on rebuild; new policy runs must first be included in the backtest-atlas index. App configuration reads only named runtime-manifest path settings from `.env` / process environment and never exports environment contents.
