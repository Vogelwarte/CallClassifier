from typing import List

from BNResultTools.RelativeTimeSegment import LabeledRelativeTimeSegment


class AudacityLabel(LabeledRelativeTimeSegment):
    """
        Represents data of a single line of the standard Audacity label file.
        The fields:

        begin time (s) <tab> end time (s) <tab> label

    """

    def __init__(self, t0: float, t1: float, label: str):
        LabeledRelativeTimeSegment.__init__(self, t0, t1, label)

    def __init__(self, raw_record: List[str]) -> None:
        begin_time = float(raw_record[0])
        end_time = float(raw_record[1])
        label = raw_record[2].strip()
        LabeledRelativeTimeSegment.__init__(self, begin_time, end_time, label)
        # SimpleLabel.__init__(label)

    # tab separated, no header line
    # 0 begin
    # 1 end
    # 2 label

    @staticmethod
    def parse_file(filename: str) -> List['AudacityLabel']:
        """
        Reads a single label file into a list of records
        :param filename: full path of the file to read
        :return:
        """
        records = []
        file = open(filename, "r")
        curr = 0
        for line in file.readlines():
            curr += 1
            data = line.split("\t")
            for i in range(0, len(data)):
                data[i] = data[i].strip()
            current = AudacityLabel(data)
            records.append(current)
        return records
