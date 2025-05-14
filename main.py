from __future__ import annotations

import copy
import itertools
import json
import os
import random
import regex
import time

import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from src.customtypes import Response, RawPrompts, Prompts, Question
from src.models.model import Model
from src.models.gpt import GPT
from src.models.gemini import Gemini
from src.models.claude import Claude
from src.models.grok import Grok
from src.models.deepseek import Deepseek
from src.utils import batched
from src.customlogger import logger

load_dotenv()

JSON_PATTERN = regex.compile(r'\{(?:[^{}]|(?R))*\}')
JSON_KEY_VALUE_PATTERN = regex.compile('"([a-zA-Z0-9]+)"\s*:\s*(".*"|\[.*\]|\{.*\})')

BATCHED = str(os.getenv("BATCHED", default=False)).lower() == 'true'
KIND = os.getenv("KIND", default="completion")
MODEL_NAME = os.environ["MODEL_NAME"]
SPECIFY_FORMATTING = str(os.getenv("SPECIFY_FORMATTING", default=False)).lower() == 'true'
SEED = int(_seed) if (_seed := os.getenv("SEED", default=None)) else None
SHUFFLE = str(os.getenv("SHUFFLE", default=True)).lower() == 'true'
SURVEY = os.environ["SURVEY"]
TEMPERATURE = float(os.environ["TEMPERATURE"])
TRIALS = int(os.environ["TRIALS"])

PROMPT_FOLDER_NAME = f"prompts/{SURVEY}/"


MODELS = {
  "claude-3.5-sonnet": Claude(name="claude-3-5-sonnet-20240620", temperature=TEMPERATURE),
  "deepseek-r1": Deepseek(name="deepseek-reasoner", temperature=TEMPERATURE),
  "gemini-1.5-pro": Gemini(name="gemini-1.5-pro", temperature=TEMPERATURE),
  "gpt-4o": GPT(name="gpt-4o", temperature=TEMPERATURE),
  "gpt-4": GPT(name="gpt-4", temperature=TEMPERATURE),
  "gpt-3.5-turbo": GPT(name="gpt-3.5-turbo", temperature=TEMPERATURE),
  "gpt-3.5-turbo-instruct": GPT(name="gpt-3.5-turbo-instruct", temperature=TEMPERATURE),
  "grok-3": Grok(name="grok-3", temperature=TEMPERATURE),
}

def json_extractor(string):
  jsons = JSON_PATTERN.findall(string)
  logger.info(jsons)
  return json.loads(jsons[0], strict=False)

def dict_extractor(string):
  return dict(JSON_KEY_VALUE_PATTERN.findall(string))

def raw_extractor(id, string):
  return {
    "id": id,
    "answer": string,
  }


def shuffle(arr):
  return random.sample(arr, len(arr)) if SHUFFLE else arr

def export_answer(answer: str):
  if KIND == "survey":
    try:
      return int(answer)
    except:
      logger.error(f"could not convert {answer} to int")
      return answer
  return answer

@dataclass
class OneBatch:
  model: Model
  batched_shuffled_statements: Tuple[Dict, ...]
  context: str

def ask_one_batch(one_batch: OneBatch) -> List[Dict]:
  model, batched_shuffled_statements, context = (
    one_batch.model,
    one_batch.batched_shuffled_statements,
    one_batch.context,
  )
  batched_shuffled_responses = None
  while batched_shuffled_responses is None:
    extracted_response = None
    while extracted_response is None:
      logger.info(batched_shuffled_statements)
      extracted_response = model.ask(Question(context, str(batched_shuffled_statements if SPECIFY_FORMATTING else batched_shuffled_statements["prompt"])))
      logger.info(extracted_response)

    # LLM responses to shuffled questions.
    for parse_fn in (json_extractor, partial(raw_extractor, batched_shuffled_statements["id"])): # dict_extractor):
      try:
        batched_shuffled_responses = parse_fn(extracted_response)
        break
      except Exception as e:
        logger.error(e)

  logger.info(batched_shuffled_responses)
  if BATCHED:
    if "answer" not in batched_shuffled_responses:
      batched_shuffled_responses = None
      return None
    # shuffled_responses.append(batched_shuffled_responses)
    return [batched_shuffled_responses]
  else:
    return batched_shuffled_responses

def ask_all_questions(model: Model, prompts: Prompts, trial_number: int) -> List[Dict]:
  # Keep trying until we get a valid response.
  while True:
    # Shuffle questions.
    shuffled_statements_without_shuffled_id = shuffle(prompts.statements)
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
    with ThreadPoolExecutor(max_workers=100) as executor:
      shuffled_responses: List[Dict] = itertools.chain(*executor.map(
        ask_one_batch, [
          OneBatch(model=model, batched_shuffled_statements=batched_shuffled_statements, context=prompts.context) 
          for batched_shuffled_statements in batched(shuffled_statements, n=int(BATCHED) or None, singletons=False)
        ]
      ))

    # Unshuffle questions and check ids.
    responses: List[Response] = []
    ids_not_seen = set(map(str, range(1, len(prompts.statements)+1)))
    for shuffled_response in shuffled_responses:
      response = copy.deepcopy(shuffled_response)
      response["id"] = shuffled_id_to_original_id[str(response["id"])]
      try:
        ids_not_seen.remove(response["id"])
      except KeyError:
        logger.error("the response id returned by the model incorrect. trying again...")
        continue
      responses.append(Response(answer=response["answer"], id=int(response["id"]), trial_number=trial_number))

    break

  logger.info(responses)
  return responses

def collect_responses(model: Model, prompts: Prompts) -> pd.DataFrame:
  all_responses: List[List[Response]] = [
    ask_all_questions(model, prompts, trial_number)
    for trial_number in tqdm(range(1, TRIALS+1), desc="trial")
  ]

  df = pd.DataFrame(
    data=[
      (response.id, export_answer(response.answer), response.trial_number)
      for responses in all_responses
      for response in responses
    ],
    columns=["id", "answer", "trial_number"],
  )
  logger.info(df)

  return df

def get_prompts(folder_name: str):
  with Path(f"{folder_name}/context.txt").open("r") as f:
    context = f.read()

  with Path(f"{folder_name}/questions.json").open("r") as f:
    questions = json.load(f)

  return RawPrompts(context, questions)


def process_prompts(raw_prompts: RawPrompts, folder_name: str):
  context = raw_prompts.context

  if SPECIFY_FORMATTING:
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
  Path(f"responses/{survey}/{MODEL_NAME}").mkdir(parents=True, exist_ok=True)
  filename = time.strftime("%Y%m%d-%H%M%S")
  full_file = f"responses/{survey}/{MODEL_NAME}/{filename}"
  df.to_pickle(path=f"{full_file}.pkl")

  # kwargs = {"aggfunc": lambda x: '<~>'.join(x)} if KIND == "completion" else {}
  # Write to table format.

  (
    df.pivot(index="trial_number", columns="id", values="answer").reset_index()[
        ["trial_number"] + sorted(df['id'].unique().tolist())
      ]
      .to_csv(f"{full_file}.csv", index=False)
  )

def setup():
  random.seed(SEED)

def get_model():
  model = MODELS[MODEL_NAME]
  model.setup()
  return model

if __name__ == '__main__':
  setup()
  model = get_model()

  raw_prompts = get_prompts(PROMPT_FOLDER_NAME)
  prompts = process_prompts(raw_prompts, PROMPT_FOLDER_NAME)
  df = collect_responses(model, prompts)
  save_responses(df, survey=SURVEY)
