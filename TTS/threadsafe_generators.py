from typing import Union, Iterator, Callable, Optional
import threading
from dataclasses import dataclass, field

@dataclass
class CharIterator:

    log_characters: bool = False
    on_character: Optional[Callable[[str], None]] = None
    on_first_text_chunk: Optional[Callable[[], None]] = None
    on_last_text_chunk: Optional[Callable[[], None]] = None

    items: list = field(default_factory=list)
    _index: int = 0
    _char_index: Optional[int] = None
    _current_iterator: Optional[Iterator[str]] = None
    immediate_stop: threading.Event = field(default_factory=threading.Event)
    iterated_text: str = ""
    first_chunk_received: bool = False

    def add(self, item: Union[str, Iterator[str]]) -> None:

        self.items.append(item)

    def stop(self) -> None:

        self.immediate_stop.set()

    def __iter__(self) -> "CharIterator":

        return self

    def _log_and_trigger(self, char: str) -> None:

        self.iterated_text += char
        if self.log_characters:
            print(char, end="", flush=True)
        if self.on_character:
            self.on_character(char)
        if not self.first_chunk_received and self.on_first_text_chunk:
            self.on_first_text_chunk()
            self.first_chunk_received = True

    def __next__(self) -> str:

        if self.immediate_stop.is_set():
            raise StopIteration

        while self._index < len(self.items):
            item = self.items[self._index]

            if isinstance(item, str):
                if self._char_index is None:
                    self._char_index = 0

                if self._char_index < len(item):
                    char = item[self._char_index]
                    self._char_index += 1
                    self._log_and_trigger(char)
                    return char
                else:
                    self._char_index = None
                    self._index += 1

            else:  
                if self._current_iterator is None:
                    self._current_iterator = iter(item)

                if self._char_index is None:
                    try:
                        self._current_str = next(self._current_iterator)
                        if hasattr(self._current_str, "choices"):
                            self._current_str = str(self._current_str.choices[0].delta.content) or ""
                    except StopIteration:
                        self._char_index = None
                        self._current_iterator = None
                        self._index += 1
                        continue

                    self._char_index = 0

                if self._char_index < len(self._current_str):
                    char = self._current_str[self._char_index]
                    self._char_index += 1
                    self._log_and_trigger(char)
                    return char
                else:
                    self._char_index = None

        if self.iterated_text and self.on_last_text_chunk:
            self.on_last_text_chunk()

        raise StopIteration

class AccumulatingThreadSafeGenerator:

    def __init__(self, gen_func: Iterator[str], on_first_text_chunk: Optional[Callable[[], None]] = None, on_last_text_chunk: Optional[Callable[[], None]] = None):

        self.lock = threading.Lock()
        self.generator = gen_func
        self.exhausted = False
        self.iterated_text = ""
        self.on_first_text_chunk = on_first_text_chunk
        self.on_last_text_chunk = on_last_text_chunk
        self.first_chunk_received = False

    def __iter__(self) -> "AccumulatingThreadSafeGenerator":

        return self

    def __next__(self) -> str:

        with self.lock:
            try:
                token = next(self.generator)
                self.iterated_text += str(token)

                if not self.first_chunk_received and self.on_first_text_chunk:
                    self.on_first_text_chunk()

                self.first_chunk_received = True
                return token

            except StopIteration:
                if self.iterated_text and self.on_last_text_chunk:
                    self.on_last_text_chunk()
                self.exhausted = True
                raise

    def is_exhausted(self) -> bool:

        with self.lock:
            return self.exhausted

    def accumulated_text(self) -> str:

        with self.lock:
            return self.iterated_text