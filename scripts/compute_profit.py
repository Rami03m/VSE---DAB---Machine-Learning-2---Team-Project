from pathlib import Path
import json
from src.io_utils import load_config, save_metrics
from src.profit import BusinessParams, argmax_profit_numeric, delta_profit

def main():
    cfg = load_config()
    p = BusinessParams(
        m_operations=cfg['business']['m_operations'],
        k=cfg['business']['k'],
        alpha=cfg['business']['alpha'],
        epsilon=cfg['business']['epsilon'],
        b=cfg['business']['b'],
    )
    # Example: read A0/A1 from saved metrics or enter manually
    A0 = 0.60  # TODO: replace with your baseline accuracy
    A1 = 0.65  # TODO: replace with your extended-model accuracy

    m0, pi0 = argmax_profit_numeric(A0, p)
    m1, pi1 = argmax_profit_numeric(A1, p)
    out = {
        "A0": A0, "A1": A1,
        "m_profit_star_A0": m0, "profit_star_A0": pi0,
        "m_profit_star_A1": m1, "profit_star_A1": pi1,
        "delta_profit": pi1 - pi0
    }
    out_path = Path(cfg['paths']['metrics_dir']) / "profit_summary.json"
    save_metrics(out, out_path)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()