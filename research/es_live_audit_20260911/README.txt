ES collected-data audit, frozen September 11 2026 22:02 ET

Open audit_report.html for the complete analysis, charts, and interactive explorer of all 87 retained setups and 772 model associations. It works offline. Linked CSV and Markdown appendices are included in this package.

Primary event metrics use ideal next-bar-open entry, fixed 15/30/60/100-minute exits, and one ES point total round-trip cost. They are hypothetical independent event outcomes, not brokerage P&L. Model-availability sensitivity uses each model's recorded timestamp. Model all-bar diagnostics separately use the repository session cost schedule and are not qualified setup trades.

Read validation_findings.md for independent arithmetic checks. 27,761 comparisons passed. See snapshot_manifest.json for source cutoff; audited_source_hashes.json captures relevant application-code versions without modifying the workspace's existing changes.

The portable ZIP omits the 916 MB frozen SQLite snapshot and Python cache. The original snapshot remains beside the report in the project workspace. Re-running source extraction is unnecessary. Full re-execution of extraction/model diagnostics requires that snapshot; the CSVs retain detailed results and source 5-minute bars for independent outcome verification.
