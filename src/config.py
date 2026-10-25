from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class OptimizationConfig:
    epoch: int = 100
    pop_size: int = 20

@dataclass
class ProjectConfig:
    # Execution flags
    apply_standardization: bool = True
    apply_hyperparam_opt: bool = True
    apply_data_cleaning: bool = False
    apply_imputation: bool = True  # should be `not apply_data_cleaning` essentially
    apply_feature_selection: bool = True

    # Model/Data params
    num_features_to_select: int = 9
    random_state: int = 42
    dataset_path: Path = PROJECT_ROOT / "data" / "dataset.csv"
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @property
    def output_dir(self) -> Path:
        # EO-tuned and default-parameter runs are kept in separate folders
        run_name = "eo_tuned" if self.apply_hyperparam_opt else "default_params"
        path = PROJECT_ROOT / "results" / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_dir(self) -> Path:
        # The EO-tuned models are the ones served by the GUI (app.py reads models/)
        path = PROJECT_ROOT / "models" if self.apply_hyperparam_opt else self.output_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output_path(self, filename: str) -> Path:
        return self.output_dir / filename

    def model_path(self, filename: str) -> Path:
        return self.models_dir / filename

config = ProjectConfig()
