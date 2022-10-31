import abc
import math
from abc import ABCMeta


def to_mm_ss(seconds: float) -> str:
    """

    :param seconds:
    :return: the string representation in format *mm:ss with leading zeros
    """
    m = int(seconds / 60)
    s = seconds % 60

    ret = str(m) + ":"
    if m < 10:
        ret = "0" + ret
    if s < 10:
        ret += "0"
    ret += str(s).rstrip('.')
    return ret


class RelativeTimeSegment(metaclass=ABCMeta):
    """
        Represents a continuous not absolute time segment, in seconds
    """

    def __init__(self, t0: float, t1: float):
        if t0 <= t1:
            self.begin_time = t0
            self.end_time = t1
        else:
            self.begin_time = t1
            self.end_time = t0

    def extended(self, additional_seconds: float) -> 'RelativeTimeSegment':
        """
        Creates a new RelativeTimeSegment extendend symetrically by the 'additional_seconds' time.
        The begin_time of the new segment will be at least 0 (never negative)

        :param additional_seconds:
        :return: the new, extended RelativeTimeSegment
        """
        nb:float = self.begin_time - additional_seconds;
        if nb < 0:
            nb = 0.0
        ne: float = self.end_time + additional_seconds
        return RelativeTimeSegment(nb, ne)


    def overlaping_time(self, other: 'RelativeTimeSegment') -> float:
        """

        :param other: another segment
        :return: the length of the overlapping time, in seconds
        """
        if self.end_time <= other.begin_time:
            return 0.0
        if self.begin_time >= other.end_time:
            return 0.0
        ov_b = max(self.begin_time, other.begin_time)
        ov_e = min(self.end_time, other.end_time)
        return ov_e - ov_b

    def duration(self) -> float:
        return self.end_time - self.begin_time

    def middle(self) -> float:
        return self.end_time / 2.0 + self.begin_time / 2.0;

    def to_mmss_string(self, separator: str = "-") -> str:
        return to_mm_ss(self.begin_time) + separator + to_mm_ss(self.end_time)
    def to_int_mmss_string(self, separator: str = "-") -> str:
        return to_mm_ss(math.floor(self.begin_time)) + separator + to_mm_ss(math.ceil(self.end_time))
    def to_string(self, separator: str = "-") -> str:
        return str(self.begin_time) + separator + str(self.end_time)



class LabeledRelativeTimeSegment(RelativeTimeSegment):

    def __init__(self, t0: float, t1: float, label: str = "") -> 'LabeledRelativeTimeSegment':
        self.label = label
        RelativeTimeSegment.__init__(self, t0, t1)


