"""输入缓冲区 - 接收和暂存输入的词素序列"""
from __future__ import annotations

from typing import List, Optional


class InputBuffer:
    """工作区的输入缓冲区

    存储：
    - raw_text: 原始输入文本
    - tokens: 词素拆分后的序列
    """

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.raw_text: str = ""
        self.tokens: List[str] = []

    def receive(self, text: str, tokens: Optional[List[str]] = None) -> None:
        """接收输入：文本和可选的预拆分词素"""
        self.raw_text = text
        if tokens is not None:
            self.tokens = list(tokens)[: self.capacity]
        else:
            self.tokens = list(text)[: self.capacity]

    def is_empty(self) -> bool:
        return not self.tokens and not self.raw_text

    def clear(self) -> None:
        self.raw_text = ""
        self.tokens = []

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"InputBuffer(raw='{self.raw_text[:30]}…', tokens={self.tokens})"
