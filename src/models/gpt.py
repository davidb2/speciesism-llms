import os

import openai
import backoff


from typing import Any

from .model import Model
from ..customtypes import Question


class GPT(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self):
    openai.api_key = os.getenv("OPENAI_API_KEY")

  @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
  def _complete(self, question: Question) -> Any:
    return openai.ChatCompletion.create(
      model=self.name,
      messages=[
        { "role": "system", "content": question.context },
        { "role": "user", "content": question.statements },
      ],
      temperature=self.temperature,
    )

  def _extract_response(self, response: Any) -> str:
    return response["choices"][0]["message"]["content"]
