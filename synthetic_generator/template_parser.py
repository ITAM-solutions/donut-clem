"""
File name: template_parser
Author: Fran Moreno
Last Updated: 9/1/2025
Version: 1.0
Description: TOFILL
"""
import random
import re

from typing import Optional, Tuple


class JinjaCustomParser:
    """ TODO fill """

    def __init__(self, text):
        self._text = text

        self.segments = None
        self.name = None
        self.idx = None
        self.idx_str = None
        self.range = None
        self.range_str = None
        self.mods = None

    def get_segments(self) -> tuple:
        """
        # TODO: fill
        :return:
        """
        regex_exp = r"([a-zA-Z]+)(?:_([^_]*))?(?:_([^_]*))?(?:_(.*))?"

        matches = re.match(regex_exp, self._text)
        if matches:
            groups = list(matches.groups())
            self.idx_str = groups[1] if groups[1] else ''
            self.range_str = groups[2] if groups[2] else ''

            groups[1] = int(groups[1]) if groups[1] is not None and groups[1] != '' else None
            groups[2] = self._parse_range(groups[2])
            groups[3] = self._parse_mods(groups[3])

            self.segments = tuple(groups)
        else:
            self.segments = tuple([None] * 4)

        self.name, self.idx, self.range, self.mods = self.segments
        return self.segments

    @staticmethod
    def get_repeat_count(range_: tuple) -> Tuple[Optional[int], int]:
        """
        # TODO: fill

        :param range_:
        :return:
        """
        if not range_:
            return None, 0

        from_ = range_[0] or 0
        to = range_[1] or from_
        top = range_[2] or 0

        count = random.randint(from_, to) if to >= from_ else 0

        if count > top > 0:
            count = top

        missing = max(0, top - count)

        return count, missing

    @staticmethod
    def _parse_range(text: str) -> Optional[tuple]:
        """
        # TODO: fill

        :param text:
        :return:
        """
        if not text:
            return None
        regex_exp = r"(\d+)(?:to(\d+))?(?:fix(\d+))?"
        matches = re.match(regex_exp, text)
        values = []
        if matches:
            for i in matches.groups():
                try:
                    values.append(int(i))
                except (ValueError, TypeError):
                    values.append(None)
            return tuple(values)
        else:
            return None

    @staticmethod
    def _parse_mods(text: str) -> Optional[tuple]:
        """
        # TODO: fill

        :param text:
        :return:
        """
        return tuple(text.split("_")) if text else tuple()

    @property
    def raw(self):
        """ # TODO: fill """
        return self._text

    @property
    def root_name(self):
        """ # TODO: fill """
        root_name = self.name
        if self.idx_str is not None:
            root_name += f'_{self.idx_str}'
        if self.range_str is not None:
            root_name += f'_{self.range_str}'
        return root_name

