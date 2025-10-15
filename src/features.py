import pandas as pd

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder: replace with pre-match features (e.g., bookmaker pre-match odds, team IDs, league dummies)
    # Avoid any leakage from post-match stats.
    cols = ['Div', 'HomeTeam', 'AwayTeam']  # example; encode later
    return df[cols].copy()

def build_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder for richer, but still pre-match, signals (e.g., rolling team form, odds, rest days)
    return build_baseline_features(df)