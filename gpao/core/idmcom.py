from __future__ import annotations

from typing import Protocol

from numpy.typing import ArrayLike


class IDmCom(Protocol):
    def send(self, cmd:ArrayLike, sleep:float|None=None):
        ...
