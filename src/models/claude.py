import anthropic
import json
import os


from typing import Any, Dict

from .model import Model
from ..customtypes import Question
from ..customlogger import logger


class Claude(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self):
    self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

  def _complete(self, question: Question) -> Any:
    messages = [
      { "role": "user", "content": question.statements, },
    ]

    return self.client.messages.create(
      model=self.name,
      max_tokens=1000,
      temperature=self.temperature,
      system=question.context,
      messages=messages,
    )

  def _extract_response(self, response: anthropic.types.Message) -> str:
    logger.info(response)
    return response.content[0].text