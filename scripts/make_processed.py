from pathlib import Path
import pandas as pd
from src.io_utils import load_config, save_df, ensure_dir
from src.prep import create_target_ou25

def main():
    cfg = load_config()
    raw_dir = Path(cfg['paths']['raw_dir'])
    out_dir = Path(cfg['paths']['processed_dir'])
    ensure_dir(out_dir)

    # Example: stitch a single market-season as a demo (customize in notebooks)
    # Here we just verify the pipeline runs without crashing.
    sample = list(raw_dir.rglob("*.csv"))
    if not sample:
        print("No CSVs found under data/raw. Please unzip your data there.")
        return
    df = pd.read_csv(sample[0])
    df = create_target_ou25(df)
    save_df(df, out_dir / "sample_processed.csv")
    print(f"Wrote {out_dir / 'sample_processed.csv'}")

if __name__ == "__main__":
    main()