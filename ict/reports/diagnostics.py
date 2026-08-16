from __future__ import annotations

import pandas as pd

from ict.setups.validators import validate_ict_setup_output


def summarize_setup_output(setup_output: pd.DataFrame) -> dict[str, object]:
    """Small Phase 1 diagnostic summary for detector outputs."""

    validate_ict_setup_output(setup_output)
    fired = setup_output.loc[setup_output["fired"].fillna(False)]
    return {
        "rows": int(len(setup_output)),
        "fired_rows": int(len(fired)),
        "unique_setup_types": sorted(str(value) for value in fired["setup_type"].dropna().unique()),
        "unique_families": sorted(str(value) for value in fired["setup_family"].dropna().unique()),
    }
