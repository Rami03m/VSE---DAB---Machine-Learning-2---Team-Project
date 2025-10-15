from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_target_ou25(df: pd.DataFrame) -> pd.DataFrame:
    # Over/Under 2.5 goals from final score. Adjust if you need a pre-match proxy.
    total_goals = df['FTHG'] + df['FTAG']
    df = df.copy()
    df['OU25'] = (total_goals >= 3).astype(int)
    return df

def split_train_val_test(df: pd.DataFrame, target: str, test_size: float, val_size: float, random_state: int):
    # First split off test, then split train into train/val
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    rel_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_df, test_size=rel_val, random_state=random_state, stratify=train_df[target])
    return train_df, val_df, test_df