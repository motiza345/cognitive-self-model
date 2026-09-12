
import os, json
import numpy as np
import pandas as pd

def run_benchmark(input_csv: str, config_json: str):
    with open(config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = os.path.join("/content/cognitive-self-model", cfg.get("report_dir", "reports/M21_2_4_3_4"))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    required = ["episode_id", "split_role", "raw_q_invalid", "label_invalid", "track"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # very light baseline outputs so pipeline is up
    df["raw_q_invalid"] = df["raw_q_invalid"].clip(0,1)
    y = df["label_invalid"].astype(int).values
    p = df["raw_q_invalid"].values

    brier = float(np.mean((p - y)**2))
    eps = 1e-12
    logloss = float(-np.mean(y*np.log(np.clip(p,eps,1-eps)) + (1-y)*np.log(np.clip(1-p,eps,1-eps))))

    summary = pd.DataFrame([{
        "n": int(len(df)),
        "prevalence": float(np.mean(y)),
        "brier_raw": brier,
        "logloss_raw": logloss
    }])
    summary.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)

    manifest = {
        "milestone": cfg.get("milestone", "M21.2.4.3.4"),
        "input_csv": input_csv,
        "config_json": config_json,
        "note": "Bootstrap minimal runner created in Colab (phase-1)."
    }
    with open(os.path.join(out_dir, "m21_2_4_3_4_identifiability_audit.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "scientific_verdict.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scientific_verdict": "CONDITIONAL_SUPPORT_LIMITED",
            "rationale": ["Pipeline bootstrapped successfully; full benchmark code pending."]
        }, f, indent=2, ensure_ascii=False)

    print("[DONE] minimal benchmark outputs written to:", out_dir)
