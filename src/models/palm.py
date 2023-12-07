import json
import os

import google.generativeai as palm

from google.generativeai.types import BlockedReason

from typing import Any, Dict

from .model import Model
from ..customtypes import Question
from ..customlogger import logger


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
    logger.info(response)
    if response.last is not None: return response.last

    # Probably PaLM blocked the response
    question: Dict = json.loads(response.messages[0]['content'].replace("'", '"'), strict=False)
    question_id = question[0]["id"]
    blocked_reason: BlockedReason = response.filters[0]['reason']
    return f'{{ "id": "{question_id}", "answer": "<!!! PaLM blocked this response. Category: {blocked_reason.name} !!!>" }}'
