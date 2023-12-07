from itertools import islice
from typing import Iterable, TypeVar, Generator, Tuple

T = TypeVar('T')

def batched(iterable: Iterable[T], n: int, singletons: bool = True) -> Generator[Tuple[T, ...], None, None]:
  # batched('ABCDEFG', 3) --> ABC DEF G
  it = iter(iterable)
  while batch := list(islice(it, n)):
    if len(batch) == 1 and not singletons:
      yield batch[0]
    else:
      yield batch