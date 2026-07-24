from typing import Generic, TypeVar

T = TypeVar("T")


class Ref(Generic[T]):
    """Mutable single-value container for late-bound dependencies.
    
    Tool handlers capture a Ref object via closure. On each invocation,
    they read `ref.value` to always get the latest value.
    Usable for deps that change after construction (media_uploader, bot_engine, etc.).
    """
    def __init__(self, value: T | None = None):
        self.value: T | None = value
