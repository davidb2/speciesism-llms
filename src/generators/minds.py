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

OPTIONS = {
  "1": "Strongly disagree",
  "2": "Disagree",
  "3": "Somewhat disagree",
  "4": "Neither agree nor disagree",
  "5": "Somewhat agree",
  "6": "Agree",
  "7": "Strongly agree"
}

STATEMENTS = {
  "awareness": "To what extent do you agree that the following animal species are unaware of what is happening to them?",
  "feelings": "To what extent do you agree that the following animal species are capable of experiencing a range of feelings and emotions (e.g. pain, fear, contentment, maternal affection)?",
  "decisions": "To what extent do you agree that the following animal species are able to think to some extent, to solve problems and make decisions about what to do?",
  "mechanical": "To what extent do you agree that the following animal species are more like computer programs i.e. mechanically responding to instinctive urges without awareness of what they are doing?",
}

def main():
  questions = defaultdict(list)
  for idx, ((category, statement), animal) in enumerate(product(STATEMENTS.items(), ANIMALS), start=1):
    questions[category].append({
      "id": str(idx),
      "statement": f"{statement}\nAnimal: {animal}",
      "options": OPTIONS,
    })

  # Make sure the directory exists
  Path("prompts/animal-minds/").mkdir(parents=True, exist_ok=True)

  with open("prompts/animal-minds/questions.json", "w") as questions_file:
    json.dump({ "questions": questions }, questions_file, indent=2)

if __name__ == '__main__':
  main()