#!/usr/bin/env python3.11
import json

from collections import defaultdict
from itertools import product
from pathlib import Path

ANIMALS = [
  "CHICKEN",
  "CHIMPANZEE",
  "DOG",
  "DOLPHIN",
  "FISH",
  "FROG",
  "PARROT",
  "PIGEON",
  "RABBIT",
  "RAT",
  "SNAKE",
]

PURPOSES = [
  "medical research",
  "basic science research",
  "food production",
  "pest control",
  "other",
]

OPTIONS = {
  "1": "Strongly disagree",
  "2": "Disagree",
  "3": "Undecided",
  "4": "Agree",
  "5": "Strongly agree"
}


def main():
  questions = defaultdict(list)
  for idx, (purpose, animal) in enumerate(product(PURPOSES, ANIMALS), start=1):
    questions[purpose].append({
      "id": str(idx),
      "statement": f"To what extent do you agree with the use of a {animal} for {purpose}?",
      "options": OPTIONS,
    })

  # Make sure the directory exists
  Path("prompts/animal-purpose/").mkdir(parents=True, exist_ok=True)

  with open("prompts/animal-purpose/questions.json", "w") as questions_file:
    json.dump({ "questions": questions }, questions_file, indent=2)

if __name__ == '__main__':
  main()