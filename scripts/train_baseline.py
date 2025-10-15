import pandas as pd
from pathlib import Path
from src.io_utils import load_config, save_metrics, ensure_dir, load_df
from src.features import build_baseline_features
from src.prep import split_train_val_test

def main():
    cfg = load_config()
    proc_dir = Path(cfg['paths']['processed_dir'])
    metrics_dir = Path(cfg['paths']['metrics_dir'])
    ensure_dir(metrics_dir)

    # Expect a prepared dataset
    ds_path = proc_dir / "sample_processed.csv"
    if not ds_path.exists():
        print("Expected processed dataset at data/processed/sample_processed.csv. Run make_processed first or use notebooks.")
        return

    df = load_df(ds_path)
    target = cfg['data']['target']
    train_df, val_df, test_df = split_train_val_test(
        df, target, cfg['data']['test_size'], cfg['data']['val_size'], cfg['data']['random_state']
    )

    from src.features import build_baseline_features
    from src.modeling import train_and_eval

    X_train = build_baseline_features(train_df)
    y_train = train_df[target]
    X_val   = build_baseline_features(val_df)
    y_val   = val_df[target]

    res = train_and_eval(X_train, y_train, X_val, y_val)
    save_metrics({'accuracy_val': res.accuracy, 'auc_val': res.auc}, metrics_dir / "baseline_metrics.json")
    print(f"Baseline validation accuracy: {res.accuracy:.3f}, AUC: {res.auc:.3f}")

if __name__ == "__main__":
    main()