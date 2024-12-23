from __future__ import annotations
import typing
from typing import Iterable, Sequence
import numpy as np


class Chunks:

    @staticmethod
    def ranges_by_sizes(sizes: Iterable):
        b = 0
        for size in sizes:
            e = b + size
            yield b, e
            b = e

    @staticmethod
    def slices_by_sizes(sizes: Iterable, *seqs: Sequence):
        ranges = Chunks.ranges_by_sizes(sizes)
        assert len(seqs) > 0
        if len(seqs) == 1:
            seq = seqs[0]
            for b, e in ranges:
                yield seq[b:e]
        else:
            for b, e in ranges:
                yield tuple(seq[b:e] for seq in seqs)
