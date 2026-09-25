import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m22_1_1.run_audit import run_audit

if __name__ == "__main__":
    run_audit(ROOT / "configs/m22_1_1_specificity_audit.json", ROOT / "reports/M22_1_1")
