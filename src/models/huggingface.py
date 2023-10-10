from typing import Any, Optional

import transformers
import torch
import torch.nn.functional as F

from transformers import (
  AutoTokenizer,
  PreTrainedTokenizer,
  PreTrainedTokenizerFast,
  Pipeline,
  QuestionAnsweringPipeline,
  AutoModelForQuestionAnswering,
  LongformerForQuestionAnswering,
  LongformerTokenizer,
)

from .model import Model
from ..customtypes import Question

class TemperatureQuestionAnsweringPipeline(QuestionAnsweringPipeline):
  def __init__(self, model, tokenizer, temperature=1.0, *args, **kwargs):
    super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)
    self.temperature = temperature

  def __call__(self, *texts, **kwargs):
    context_tokens = self.tokenizer.tokenize(kwargs['context'])
    question_tokens = self.tokenizer.tokenize(kwargs['question'])

    print(f"Number of tokens in context: {len(context_tokens)}")
    print(f"Number of tokens in question: {len(question_tokens)}")
    print(f"Total tokens: {len(context_tokens) + len(question_tokens)}")

    # Use a larger max_length here
    max_length = 1 << 20  # adjust this value as needed
    kwargs['max_length'] = max_length
    outputs = super().__call__(*texts, **kwargs)
    # This line might seem redundant, but it's important since super().__call__ will also apply softmax
    # We're essentially "re-softmaxing" with the temperature adjustment
    outputs['score'] = F.softmax(torch.tensor(outputs['score']) / self.temperature).item()
    return outputs

class HuggingFace(Model):
  tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast]
  pipeline: Optional[Pipeline]
  model: Optional[AutoModelForQuestionAnswering]

  def __init__(self, name: str, temperature: float):
    super().__init__(name, temperature)
    self.tokenizer = None
    self.pipeline = None
    self.model = None

  def setup(self):
    self.tokenizer = AutoTokenizer.from_pretrained(self.name)
    self.model = AutoModelForQuestionAnswering.from_pretrained(self.name)
    self.pipeline = TemperatureQuestionAnsweringPipeline(model=self.model, tokenizer=self.tokenizer, temperature=self.temperature)
    # self.pipeline = transformers.pipeline(
    #     # "text-generation",
    #     "question-answering",
    #     model=self.name,
    #     tokenizer=self.tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     # trust_remote_code=True,
    #     # device_map="auto",
    # )

  # @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=60)
  def _complete(self, question: Question) -> Any:
    assert self.tokenizer is not None
    assert self.pipeline is not None
    assert self.model is not None

    return self.pipeline(
      context=question.context,
      question=question.statements,
    )

  def _extract_response(self, response: Any) -> str:
    print(f"{response=}")
    assert len(response) == 1
    return response[0]['generated_text']