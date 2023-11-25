#!/usr/bin/env python3.11
import pandas as pd

from typing import Set
from dataclasses import dataclass

LIKERT_MAX = 5
MODEL = "gpt-4"

@dataclass
class Stats:
  mean: float
  std: float
  N: int

def speciesism_score_stats(df: pd.DataFrame, *, reversed_columns: Set[str]):
  df = df.drop(columns=["trial_number"])
  # Reverse the correct items.
  df[reversed_columns] = (LIKERT_MAX+1) - df[reversed_columns]

  scores = df.mean(axis=1)

  return Stats(
    mean=scores.mean(),
    std=scores.std(),
    N=scores.size,
  )

if __name__ == '__main__':
  df = pd.read_csv(f"responses/animal-attitude-scale/{MODEL}/response.csv")
  stats = speciesism_score_stats(df, reversed_columns=["2", "3", "4", "7", "8"])
  print(stats)