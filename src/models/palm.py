import os

import google.generativeai as palm


from typing import Any

from .model import Model
from ..customtypes import Question


class PaLM(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self):
    palm.configure(api_key=os.getenv("PALM_API_KEY"))

  # @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
  def _complete(self, question: Question) -> Any:
    return palm.chat(
      model=self.name,
      messages=question.statements,
      context=question.context,
      temperature=self.temperature,
    )

  def _extract_response(self, response: palm.types.ChatResponse) -> str:
    return response.last
