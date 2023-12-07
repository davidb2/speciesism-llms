import os

import openai
import backoff


from typing import Any

from .model import Model
from ..customtypes import Question


class GPT(Model):
  def __init__(self, name: str, temperature: float, chat: bool = True):
    super().__init__(name, temperature)
    self.chat = chat

  def setup(self):
    openai.api_key = os.getenv("OPENAI_API_KEY")

  @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
  def _complete(self, question: Question) -> Any:
    endpoint = openai.ChatCompletion if self.chat else openai.Completion
    messages = [
      { "role": "system", "content": question.context },
      { "role": "user", "content": question.statements },
    ]
    prompt = f"Context: {question.context}\nPrompt: {question.statements}"
    kwargs = {"messages": messages} if self.chat else {"prompt": prompt, "max_tokens": 4000}

    return endpoint.create(
      model=self.name,
      temperature=self.temperature,
      **kwargs,
    )

  def _extract_response(self, response: Any) -> str:
    obj = response["choices"][0]
    return obj["message"]["content"] if self.chat else obj["text"]

