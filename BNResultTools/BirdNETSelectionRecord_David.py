import os
from pathlib import PureWindowsPath, Path
from typing import List
from datetime import date
import sys

from BNResultTools.BirdNETResultRecord import BirdNETResultRecord
from BNResultTools.RelativeTimeSegment import RelativeTimeSegment


# Standard BirdNET-Analyzer result in the  selections.table.txt default format
# It seems the format is based on the Raven annotations - see the Table1SelectionRecord.py
#
# The file consists of header line followed by 0 or more data lines.
# Every line is a sequence of tab separated values

class BirdNETSelectionRecord_David(BirdNETResultRecord):
    """
        Represents data of a single line of the standard Raven Table.1.Selections file:
        array index | name | remark
        selec	View	Channel	start	end	bottom.freq	top.freq	Species Code	Common Name	Confidence	type	selec.file	sound.files


        0 | Selection  |  the sequence number of the line, first data line is 1
        1 | View | ???
        2 | Channel
        3 | Begin Time (s)
        4 | End Time (s)
        5 | Low Freq (Hz)
        6 | High Freq (Hz)
        7 | Species Code
        8 | Common Name
        9 | Confidence
        10 | type
        11| selec.file
        12| sound.files

        "Selection" value starts with "1" (not "0")
    """
    headers = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
               "Species Code", "Common Name", "Confidence", "type", "selec.file", "sound.files"]

    def __init__(self, raw_record: List[str]) -> None:
        self.selection = int(raw_record[0])
        self.view = raw_record[1]
        self.channel = int(raw_record[2])
        begin_time = float(raw_record[3])
        end_time = float(raw_record[4])
        self.low_freq = float(raw_record[5])
        self.high_freq = float(raw_record[6])
        species_code = raw_record[7].strip()
        common_name = raw_record[8].strip()
        confidence = float(raw_record[9])
        self.type = int(raw_record[10])
        self.selec_file = (raw_record[11].strip("\""))
        self.sound_files = (raw_record[12].strip("\""))
        super().__init__(begin_time, end_time, confidence, species_code, common_name)

    # tab separated values with a header line
    # example of a file with 3 data lines:

    # Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Species Code	Common Name	Confidence
    # 1	Spectrogram 1	1	1013.0	1016.0	150	12000	rocpta1	Rock Ptarmigan	0.8846
    # 2	Spectrogram 1	1	1014.0	1017.0	150	12000	rocpta1	Rock Ptarmigan	0.4061
    # 3	Spectrogram 1	1	3060.0	3063.0	150	12000	rocpta1	Rock Ptarmigan	0.1408

    @staticmethod
    def parse_file(filename: str) -> List['BirdNETSelectionRecord_David']:
        """ Reads a single file into a list of recrods
            :param filename: full path of the file to read
            :return: list of records, possibly empty
        """
        records = []
        file = open(filename, "r")
        curr = 0
        for line in file.readlines():
            curr += 1
            data = line.split(",")
            for i in range(0, len(data) ):
                data[i] = data[i].strip()
            if curr == 1:
                labels = data
                continue
            current = BirdNETSelectionRecord_David(data)
            records.append(current)
        return records
