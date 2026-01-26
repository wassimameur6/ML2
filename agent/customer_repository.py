"""
Customer repository backed by the churn CSV.
Only allows access to a single customer by CLIENTNUM.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd


class CustomerRepository:
    def __init__(self, data_path: str) -> None:
        csv_path = os.path.join(data_path, "churn2.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Customer CSV not found at {csv_path}")
        self._csv_path = csv_path
        self._df = pd.read_csv(csv_path)

    def get_customer(self, client_num: int) -> Optional[Dict[str, Any]]:
        match = self._df[self._df["CLIENTNUM"] == client_num]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def update_email(self, client_num: int, new_email: str) -> bool:
        match = self._df[self._df["CLIENTNUM"] == client_num]
        if match.empty:
            return False
        idx = match.index[0]
        self._df.at[idx, "Email"] = new_email
        self._df.to_csv(self._csv_path, index=False)
        return True

    def update_credit_limit(self, client_num: int, new_limit: float) -> bool:
        match = self._df[self._df["CLIENTNUM"] == client_num]
        if match.empty:
            return False
        idx = match.index[0]
        self._df.at[idx, "Credit_Limit"] = new_limit
        self._df.to_csv(self._csv_path, index=False)
        return True
