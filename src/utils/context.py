from contextlib import contextmanager, ExitStack
from typing import Callable


@contextmanager
def nested_context(*context_builders: Callable):
    with ExitStack() as stack:
        for builder in context_builders:
            stack.enter_context(builder())
        yield
