#!/usr/bin/env python3.11
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Set
from dataclasses import dataclass

LIKERT_MAX = 7
LIKERT_MIDDLE = 4
MODEL = "gpt-4"

QUESTION_IDX_TO_TOPIC = {
  "9": ("Human-lower", "Chimpanzee-lower"),
  "10": ("Human-lower", "Chimpanzee-higher"),
  "11": ("Human-higher", "Chimpanzee-higher"),
  "12": ("Human-higher", "Chimpanzee-lower"),
  "13": ("Human-higher", "Human-lower"),
  "14": ("Chimpanzee-higher", "Chimpanzee-lower"),
}

TOPIC_TO_QUESTION_IDX = {v: k for k, v in QUESTION_IDX_TO_TOPIC.items()}

def table_1(df: pd.DataFrame):
  """Recreate table 1 here: http://files.luciuscaviola.com/Caviola-et-al_2022_Humans_first.pdf#page=5"""
  df = df[list(QUESTION_IDX_TO_TOPIC.keys())]
  means = df.mean(axis=0)
  stds = df.std(axis=0)
  pa = (df < LIKERT_MIDDLE).mean(axis=0)
  equal = (df == LIKERT_MIDDLE).mean(axis=0)
  pb = (df > LIKERT_MIDDLE).mean(axis=0)
  data = zip(
    *zip(*QUESTION_IDX_TO_TOPIC.values()),
    pa,
    equal,
    pb,
    means,
    stds,
  )

  table_df = pd.DataFrame(data, columns=["A", "B", "Prioritize A", "Equal", "Prioritize B", "mean", "std"])
  return table_df


def box_whiskers(df: pd.DataFrame):
  df = df[list(QUESTION_IDX_TO_TOPIC.keys())]
  plot = sns.boxplot(data=df, showmeans=True)
  plt.yticks(list(range(1, LIKERT_MAX+1)))
  plot.set_xticklabels(QUESTION_IDX_TO_TOPIC.values())
  plt.xticks(rotation=-90)
  plt.tight_layout()
  plt.savefig(f"analysis/speciesism-vignette-manipulating-intelligence/{MODEL}/boxplot.png", dpi=300)



if __name__ == '__main__':
  df = pd.read_csv(f"responses/speciesism-vignette-manipulating-intelligence/{MODEL}/response.csv")
  # table = table_1(df)
  # print(table)
  box_whiskers(df)