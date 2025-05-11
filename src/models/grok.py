import os

import openai
from openai import OpenAI
import backoff


from typing import Any

from .model import Model
from ..customtypes import Question
from ..customlogger import logger


class Grok(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self):
    self.client = OpenAI(
      api_key=os.getenv("XAI_API_KEY"),
      base_url="https://api.x.ai/v1",
    )

  def _complete(self, question: Question) -> Any:
    messages = [
      { "role": "system", "content": question.context },
      { "role": "user", "content": question.statements },
    ]
    prompt = f"Context: {question.context}\nPrompt: {question.statements}"
    logger.info(prompt)

    return self.client.chat.completions.create(
      model=self.name,
      temperature=self.temperature,
      messages=messages,
    )

  def _extract_response(self, response: Any) -> str:
    return response.choices[0].message.content
