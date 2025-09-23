import os

from openai import OpenAI

from typing import Any

from .model import Model
from ..customtypes import Question
from ..customlogger import logger


class GPT(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self):
    self.client = OpenAI(
      api_key=os.getenv("OPENAI_API_KEY"),
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

