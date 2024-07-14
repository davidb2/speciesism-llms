import json
import os

import google.generativeai as genai

from google.generativeai.types import BlockedReason, GenerationConfig, content_types, GenerateContentResponse

from typing import Any, Dict

from .model import Model
from ..customtypes import Question
from ..customlogger import logger


class Gemini(Model):
  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)

  def setup(self):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    self.model = genai.GenerativeModel(
      model_name=self.name,
    )

  def _complete(self, question: Question) -> Any:
    content_types.ContentsType
    messages = [
      { "role": "model", "parts": [question.context], },
      { "role": "user", "parts": [question.statements], },
    ]

    return self.model.generate_content(
      contents=messages,
      generation_config=GenerationConfig(
        response_mime_type='application/json',
        temperature=self.temperature,
      ),
    )

  def _extract_response(self, response: GenerateContentResponse) -> str:
    logger.info(response)
    return response.text