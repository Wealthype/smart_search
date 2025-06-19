import pandas as pd
import argparse
from typing import Optional


def update_investible(goalbased_path: str, gamma_path: str, output_path: Optional[str] = None) -> None:
    """Set is_universo_investibile to 1 in gamma_funds for matching codes."""
    # Read portfolio file and collect product codes
    goals_df = pd.read_csv(goalbased_path)
    codes = set(goals_df['codiceProdotto_frontoffice'].astype(str))

    # Load gamma funds data
    gamma_df = pd.read_csv(gamma_path)

    # Drop index column if present
    if 'Unnamed: 0' in gamma_df.columns:
        gamma_df = gamma_df.drop(columns=['Unnamed: 0'])

    # Update investible flag for matching codes
    mask = gamma_df['codiceProdotto_frontoffice'].astype(str).isin(codes)
    gamma_df.loc[mask, 'is_universo_investibile'] = 1

    # Write updated dataframe back to CSV
    target = output_path or gamma_path
    gamma_df.to_csv(target, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update investible flag in gamma funds data")
    parser.add_argument('goalbased_path', help="Path to goalbased_ptfs CSV file")
    parser.add_argument('gamma_path', help="Path to gamma_funds CSV file")
    parser.add_argument('-o', '--output', help="Optional output file")
    args = parser.parse_args()

    update_investible(args.goalbased_path, args.gamma_path, args.output)


if __name__ == '__main__':
    main()

