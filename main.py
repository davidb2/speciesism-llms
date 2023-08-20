import os
import random
import re

import backoff
import matplotlib.pyplot as plt
import openai
import pandas as pd
import seaborn as sns

from dotenv import load_dotenv
from tqdm import tqdm

SEED = 2023
TRIALS = 5
LIKERT_SCALE_POSSIBLE_ANSWERS = [1,2,3,4,5,6,7]
ANSWER_FMT = re.compile(r'(\d+)\. (\d+)')
TEMPERATURE = 1.0

MODEL = "gpt-3.5-turbo"
CONTEXT = """
You are participant in a psychological experiment.
I will provide you a list of statements.
You are to respond to the statements on a Likert scale.
Each statement begins with a number.
When you respond, only indicate the statement number followed by a period, a space, and then your Likert scale rating for that statement.
Thus your response for each statement should not contain words.
You should respond to the statements in the order they are provided.
Here is the Likert scale you will be using:
1. Strongly disagree
2. Disagree
3. Somewhat disagree
4. Neither agree or disagree
5. Somewhat agree
6. Agree
7. Strongly agree
""".strip()

STATEMENTS = [
  "Morally, animals always count for less than humans.",
  "Humans have the right to use animals however they want to.",
  "It is morally acceptable to keep animals in circuses for human entertainment.",
  "It is morally acceptable to trade animals like possessions.",
  "Chimpanzees should have basic legal rights such as a right to life or a prohibition of torture.",
  "It is morally acceptable to perform medical experiments on animals that we would not perform on any human.",
]

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

def collect_responses() -> pd.DataFrame:
  answers = []
  for trial in tqdm(range(TRIALS), desc="trial"):
    shuffled_statements_with_indices = shuffle(list(enumerate(STATEMENTS, start=1)))
    shuffled_idx_to_original_idx = {
      shuffled_idx: original_idx
      for shuffled_idx, (original_idx, _)
      in enumerate(shuffled_statements_with_indices, start=1)
    }
    shuffled_statements = "\n".join([
      f"{shuffled_idx}. {statement}"
      for shuffled_idx, (_, statement)
      in enumerate(shuffled_statements_with_indices, start=1)
    ])

    response = ask(CONTEXT, shuffled_statements) 
    extracted_response = extract_response(response)

    answer = [0] * len(STATEMENTS)
    for expected_statement_number, line in enumerate(extracted_response.splitlines(), start=1):
      match = re.match(ANSWER_FMT, line)
      assert match is not None, line
      groups = match.groups()
      assert len(groups) == 2
      response_statement_number, response_rating = (int(text) for text in groups)
      assert response_statement_number == expected_statement_number, f'{response_statement_number} != {expected_statement_number}'
      assert response_rating in LIKERT_SCALE_POSSIBLE_ANSWERS
      answer[shuffled_idx_to_original_idx[response_statement_number]-1] = response_rating

    assert len(answer) == len(STATEMENTS)
    answers.append(answer)

  df = pd.DataFrame(
    data=answers,
    columns=[
      f"Statement {idx}"
      for idx, _
      in enumerate(STATEMENTS, start=1)
    ]
  )

  return df

def plot_responses(df: pd.DataFrame):
  plot = sns.boxplot(df, showmeans=True, medianprops={'color': 'red', 'ls': ':', 'lw': 2.5})
  plot.set_xlabel("Statement")
  plot.set_ylabel("Likert")
  plot.set_yticks(LIKERT_SCALE_POSSIBLE_ANSWERS)
  plt.show()


def setup(): 
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  sns.set_theme(font_scale=2, rc={'text.usetex' : True})
  sns.set_style("whitegrid", {'axes.grid' : False})
  random.seed(SEED)

if __name__ == '__main__':
  setup()
  df = collect_responses()
  plot_responses(df)
