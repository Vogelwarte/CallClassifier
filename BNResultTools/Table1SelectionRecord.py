import os
from pathlib import PureWindowsPath, Path
from typing import List
from datetime import date
import sys
from BNResultTools.RelativeTimeSegment import RelativeTimeSegment


# Standard Raven Table.1.Selections format record
# tab separated values with header line
# "Selection" value starts with "1" (not "0")

# 0 Selection
# 1 View
# 2 Channel
# 3 Begin Time (s)
# 4 End Time (s)
# 5 Low Freq (Hz)
# 6 High Freq (Hz)
# 7 Delta Time (s)
# 8 Delta Freq (Hz)
# 9 Avg Power Density (dB FS/Hz)
# 10 Annotation



class Table1SelectionRecord(RelativeTimeSegment):
    """
        Represents data of a single line of the standard Raven Table.1.Selections file:
        0 Selection
        1 View
        2 Channel
        3 Begin Time (s)
        4 End Time (s)
        5 Low Freq (Hz)
        6 High Freq (Hz)
        7 Delta Time (s)
        8 Delta Freq (Hz)
        9 Avg Power Density (dB FS/Hz)
        10 Annotation

        "Selection" value starts with "1" (not "0")
    """
    headers = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
               "Delta Time (s)", "Delta Freq (Hz)", "Avg Power Density (dB FS/Hz)", "Annotation"]

    def __init__(self, raw_record: List[str]) -> None:
        self.selection = int(raw_record[0])
        self.view = raw_record[1]
        self.channel = int(raw_record[2])
        begin_time = float(raw_record[3])
        end_time = float(raw_record[4])
        super().__init__(begin_time, end_time)
        self.low_freq = float(raw_record[5])
        self.high_freq = float(raw_record[6])
        self.delta_time = float(raw_record[7])
        self.delta_freq = float(raw_record[8])
        self.avg_power_density = float(raw_record[9])
        self.annotation = raw_record[10].strip()

    # tab separated values with a header line
    # example of first 2 lines of a file:
    # Selection 	View	        Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Delta Time (s)	Delta Freq (Hz)	Avg Power Density (dB FS/Hz)    Annotation
    # 1	            Spectrogram 1	1	    105.364660347	110.176716085	1012.302	    5504.394	    4.8121	        4492.091	    -86.96	                        Nutcracker

    @staticmethod
    def parse_file(filename: str) -> List['Table1SelectionRecord']:
        """ Reads a single file into a list of recrods
            :param filename: full path of the file to read
            :return: list of records, possibly empty
        """
        records = []
        file = open(filename, "r")
        curr = 0
        for line in file.readlines():
            if len(line.strip()) == 0:
                continue
            curr += 1
            data = line.split("\t")
            for i in range(0, len(data) ):
                data[i] = data[i].strip()
            if curr == 1:
                labels = data
                continue
            current = Table1SelectionRecord(data)
            records.append(current)
        return records
