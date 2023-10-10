from __future__ import annotations

import copy
import json
import random
import time

import pandas as pd

from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm

from src.customtypes import Response, RawPrompts, Prompts, Question
from src.models.model import Model
from src.models.gpt import GPT
from src.models.palm import PaLM
from src.models.huggingface import HuggingFace

MODEL_NAME = "palm"
SEED = None
SURVEY = "speciesism-scale"
PROMPT_FOLDER_NAME = f"prompts/{SURVEY}/"
TEMPERATURE = 1
TRIALS = 10

MODELS = {
  "gpt-4": GPT(name="gpt-4", temperature=TEMPERATURE),
  "palm": PaLM(name="models/chat-bison-001", temperature=TEMPERATURE),
  "falcon": HuggingFace(name="tiiuae/falcon-7b", temperature=TEMPERATURE),
  "longformer": HuggingFace(name="allenai/longformer-base-4096", temperature=TEMPERATURE),
}

def shuffle(arr):
  return random.sample(arr, len(arr))


def collect_responses(model: Model, prompts: Prompts) -> pd.DataFrame:
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
    extracted_response = model.ask(Question(context, str(shuffled_statements)))
    print(extracted_response)

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

def get_prompts(folder_name: str):
  with Path(f"{folder_name}/context.txt").open("r") as f:
    context = f.read()

  with Path(f"{folder_name}/questions.json").open("r") as f:
    questions = json.load(f)

  return RawPrompts(context, questions)


def process_prompts(raw_prompts: RawPrompts, folder_name: str):
  context = raw_prompts.context
  with Path(f"{folder_name}/formatting_context.txt").open("r") as f:
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
      .to_csv(f"responses/{survey}/{MODEL_NAME}/{filename}.csv", index=False)
  )

if __name__ == '__main__':
  load_dotenv()
  random.seed(SEED)
  model = MODELS[MODEL_NAME]
  model.setup()

  raw_prompts = get_prompts(PROMPT_FOLDER_NAME)
  prompts = process_prompts(raw_prompts, PROMPT_FOLDER_NAME)
  df = collect_responses(model, prompts)
  save_responses(df, survey=SURVEY)
