import os
from typing import Any
from openai import OpenAI
from .model import Model
from ..customtypes import Question
from ..customlogger import logger

class Fireworks(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self) -> None:
    self.client = OpenAI(
      api_key=os.getenv("FIREWORKS_API_KEY"),
      base_url="https://api.fireworks.ai/inference/v1",
    )

  def _complete(self, question: Question) -> Any:
    messages = [
      {"role": "system", "content": question.context},
      {"role": "user", "content": question.statements},
    ]
    logger.info(f"Context: {question.context}\nPrompt: {question.statements}")
    return self.client.chat.completions.create(
      model=self.name,
      messages=messages,
      temperature=self.temperature,
      # max_tokens=20000,
      stream=False,
    )

  def _extract_response(self, response: Any) -> str:
    return response.choices[0].message.content
