from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ICTArtifactLayout:
    root: Path
    phase01_scan: Path
    phase02_features: Path
    phase03_labeling: Path
    phase04_prepared: Path
    phase05_training: Path
    phase06_thresholds: Path
    phase07_backtests: Path

    def ensure_directories(self) -> "ICTArtifactLayout":
        for path in self.as_dict().values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, Path]:
        return {
            "root": self.root,
            "phase01_scan": self.phase01_scan,
            "phase02_features": self.phase02_features,
            "phase03_labeling": self.phase03_labeling,
            "phase04_prepared": self.phase04_prepared,
            "phase05_training": self.phase05_training,
            "phase06_thresholds": self.phase06_thresholds,
            "phase07_backtests": self.phase07_backtests,
        }


def build_ict_artifact_layout(
    run_id: str,
    *,
    base_dir: str | Path = "artifacts",
    ensure_directories: bool = False,
) -> ICTArtifactLayout:
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise ValueError("run_id must be a non-empty string.")

    root = Path(base_dir) / normalized_run_id
    layout = ICTArtifactLayout(
        root=root,
        phase01_scan=root / "phase01_scan",
        phase02_features=root / "phase02_features",
        phase03_labeling=root / "phase03_labeling",
        phase04_prepared=root / "phase04_prepared",
        phase05_training=root / "phase05_training",
        phase06_thresholds=root / "phase06_thresholds",
        phase07_backtests=root / "phase07_backtests",
    )
    return layout.ensure_directories() if ensure_directories else layout
