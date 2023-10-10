from dataclasses import dataclass
from typing import List, Dict

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

@dataclass
class Question:
  context: str
  statements: str