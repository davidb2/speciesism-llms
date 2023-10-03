from __future__ import annotations

import copy
import json
import os
import random
import time

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

MODEL = "gpt-4"
SEED = 20
SURVEY = "speciesism-prioritization-tasks"
TEMPERATURE = 1
TRIALS = 10

@dataclass
class Response:
  answer: str
  id: int
  trial_number: int

@dataclass
class RawPrompts:
  context: str
  statements: Dict

@dataclass
class Prompts:
  context: str
  statements: Dict

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


def collect_responses(prompts: Prompts) -> pd.DataFrame:
  context = prompts.context
  statements = prompts.statements
  all_responses: List[List[Response]] = []

  # TODO: make this loop resilient to LLM format errors while making sure
  # it does not use up LLM quota. Currently one little error would ruin the entire experiment,
  # which could mean a couple of minutes went to waste.
  for trial_number in tqdm(range(1, TRIALS+1), desc="trial"):
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
    print(shuffled_responses)

    # Unshuffle questions and check ids.
    responses: List[Response] = []
    ids_not_seen = set(map(str, range(1, len(statements)+1)))
    for shuffled_response in shuffled_responses:
      response = copy.deepcopy(shuffled_response)
      response["id"] = shuffled_id_to_original_id[response["id"]]
      ids_not_seen.remove(response["id"])
      responses.append(Response(answer=response["answer"], id=int(response["id"]), trial_number=trial_number))

    print(responses)
    all_responses.append(responses)


  df = pd.DataFrame(
    data=[
      (response.id, int(response.answer), response.trial_number)
      for responses in all_responses
      for response in responses
    ],
    columns=["id", "answer", "trial_number"],
  )

  return df

def plot_responses(df: pd.DataFrame):
  plot = sns.boxplot(
    df,
    x="id",
    y="answer",
    showmeans=True,
    medianprops={'color': 'red', 'ls': ':', 'lw': 2.5}
  )
  plot.set_xlabel("Statement Id")
  plot.set_ylabel("Answer")
  plt.savefig(f'plots/result.png', dpi=300, bbox_inches="tight")
  plt.show()

def setup(): 
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  sns.set_theme(font_scale=2, rc={'text.usetex' : False})
  sns.set_style("whitegrid", {'axes.grid' : False})
  random.seed(SEED)


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

def save_responses(df: pd.DataFrame, *, survey: str):
  # Make sure the directory exists
  Path(f"responses/{survey}").mkdir(parents=True, exist_ok=True)
  
  # Write to table format.
  filename = time.strftime("%Y%m%d-%H%M%S")

  (
    df.pivot_table(index="trial_number", columns="id", values="answer").reset_index()[
        ["trial_number"] + sorted(df['id'].unique().tolist())
      ]
      .to_csv(f"responses/{survey}/{filename}.csv")
  )

if __name__ == '__main__':
  setup()
  raw_prompts = get_prompts(f"prompts/{SURVEY}/")
  prompts = process_prompts(raw_prompts)
  df = collect_responses(prompts)
  save_responses(df, survey=SURVEY)
