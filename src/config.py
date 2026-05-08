from pathlib import Path

RANDOM_STATE = 42

# Project folder: where train.py exists
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset folder on your computer
DATA_DIR = Path(r"E:\DataScience_project")

DEVELOPMENT_CSV = DATA_DIR / "development.csv"
EVALUATION_CSV = DATA_DIR / "evaluation.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

DEV_AUDIO_CANDIDATES = [
    DATA_DIR / "audios_development",
    DATA_DIR / "audio_development",
]

EVAL_AUDIO_CANDIDATES = [
    DATA_DIR / "audios_evaluation",
    DATA_DIR / "audio_evaluation",
]

DEVELOPMENT_AUDIO_DIR = next(
    (p for p in DEV_AUDIO_CANDIDATES if p.exists()),
    DEV_AUDIO_CANDIDATES[0]
)

EVALUATION_AUDIO_DIR = next(
    (p for p in EVAL_AUDIO_CANDIDATES if p.exists()),
    EVAL_AUDIO_CANDIDATES[0]
)

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = BASE_DIR / "cache"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)