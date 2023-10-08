from abc import ABC, abstractmethod
from typing import Any

class Question:
  context: str
  statements: str


class Model(ABC):
  name: str
  temperature: float

  def __init__(self, name: str, temperature: float):
    super().__init__()
    self.name = name
    self.temperature = temperature

  def ask(self, question: Question) -> str:
    """Ask the model a question and return an answer as a string."""
    completion = self._complete(question)
    return self._extract_response(completion)

  @abstractmethod
  def setup(self) -> None:
    """API Keys"""

  @abstractmethod
  def _complete(self, question: Question) -> Any:
    """Get a response from the model"""

  @abstractmethod
  def _extract_response(self, response: Any) -> str:
    """Turn the response into a string."""