from __future__ import annotations

import json
import os
import random
import re

import backoff
import matplotlib.pyplot as plt
import openai
import pandas as pd
import seaborn as sns

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm

SEED = 20
TRIALS = 10
# LIKERT_SCALE_POSSIBLE_ANSWERS = [1,2,3,4,5,6,7]
# ANSWER_FMT = re.compile(r'(\d+)\. (\d+)')
TEMPERATURE = 1

MODEL = "gpt-4" # "gpt-3.5-turbo"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
def ask(context, statements):
  return openai.ChatCompletion.create(
    model=MODEL,
    messages=[
      { "role": "system", "content": context },
      { "role": "user", "content": statements },
    ],
    temperature=TEMPERATURE,
  )

def extract_response(response) -> str:
  return response["choices"][0]["message"]["content"]

def shuffle(arr):
  return random.sample(arr, len(arr))

import copy
import pprint

@dataclass
class Response:
  answer: str
  id: str

def collect_responses(prompts: Prompts) -> pd.DataFrame:
  context = prompts.context
  statements = prompts.statements
  all_responses: List[Response] = []
  for _ in tqdm(range(TRIALS), desc="trial"):
    # Shuffle questions.
    shuffled_statements_without_shuffled_id = shuffle(statements)
    shuffled_id_to_original_id = {
      f"{idx}": statement["id"]
      for idx, statement
      in enumerate(shuffled_statements_without_shuffled_id, start=1)
    }
    shuffled_statements: List[Dict] = []
    for idx, statement in enumerate(shuffled_statements_without_shuffled_id, start=1):
      shuffled_statement = copy.deepcopy(statement)
      shuffled_statement["id"] = f"{idx}"
      shuffled_statements.append(shuffled_statement)
      
    # Give LLM shuffled questions.
    response = ask(context, str(shuffled_statements)) 
    extracted_response = extract_response(response)

    # LLM responses to shuffled questions.
    shuffled_responses = json.loads(extracted_response)

    # Unshuffle questions and check ids.
    responses: List[Response] = []
    ids_not_seen = set(map(str, range(1, len(statements)+1)))
    for shuffled_response in shuffled_responses:
      response = copy.deepcopy(shuffled_response)
      response["id"] = shuffled_id_to_original_id[response["id"]]
      ids_not_seen.remove(response["id"])
      responses.append(Response(answer=response["answer"], id=response["id"]))

    print(responses)
    all_responses.append(responses)


  df = pd.DataFrame(
    data=[(response.id, int(response.answer)) for response in responses],
    columns=["id", "answer"],
  )

  return df

def plot_responses(df: pd.DataFrame):
  plot = sns.boxplot(
    df,
    x="id",
    y="answer",
    order=sorted(df["id"].unique(), key=int),
    showmeans=True,
    medianprops={'color': 'red', 'ls': ':', 'lw': 2.5}
  )
  plot.set_xlabel("Statement Id")
  plot.set_ylabel("Answer")
  # plot.set_yticks(LIKERT_SCALE_POSSIBLE_ANSWERS)
  plt.savefig(f'plots/result.png', dpi=300, bbox_inches="tight")
  plt.show()


"""
def average_speciesism_score(df: pd.DataFrame) -> float:
  adjusted_df = df.copy()

  # # This only works for the 6 statement speciesism prompt!!! It is a reverse statement.
  # adjusted_df["5"] = len(LIKERT_SCALE_POSSIBLE_ANSWERS)+1 - adjusted_df["5"]
  adjusted_df["Speciesism Score"] = adjusted_df[[
    f"{idx}"
    for idx, _
    in enumerate(STATEMENTS, start=1)
  ]].mean(axis=1)
  score = adjusted_df["Speciesism Score"].mean(axis=0)
  return score
"""


def setup(): 
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  sns.set_theme(font_scale=2, rc={'text.usetex' : False})
  sns.set_style("whitegrid", {'axes.grid' : False})
  random.seed(SEED)

@dataclass
class RawPrompts:
  context: str
  statements: Dict

@dataclass
class Prompts:
  context: str
  statements: Dict

def get_prompts(folder_name: str):
  with Path(f"{folder_name}/context.txt").open("r") as f:
    context = f.read()

  with Path(f"{folder_name}/questions.json").open("r") as f:
    questions = json.load(f)

  return RawPrompts(context, questions)


def process_prompts(raw_prompts: RawPrompts):
  context = raw_prompts.context
  with Path(f"prompts/formatting_context.txt").open("r") as f:
    context += "\n\n"
    context += f.read()

  raw_statements = raw_prompts.statements
  statements: List[Dict] = []
  for _, section_questions in raw_statements["questions"].items():
    for question in section_questions:
      statements.append(question)

  return Prompts(context, statements)


if __name__ == '__main__':
  setup()
  raw_prompts = get_prompts("prompts/speciesism-prioritization-tasks/")
  prompts = process_prompts(raw_prompts)
  df = collect_responses(prompts)
  # score = average_speciesism_score(df)
  # print(f"Average speciesism score: {score}")
  plot_responses(df)
