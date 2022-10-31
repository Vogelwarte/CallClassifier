import os
from pathlib import PureWindowsPath, Path
from typing import List
from datetime import date
import sys

from BNResultTools.BirdNETResultRecord import BirdNETResultRecord


class BnCsvResultRecord(BirdNETResultRecord):
    """
        Represents data of a single line of the standard BirdNet-analyzer csv result file.
        The fields:

        Selection | View | Channel | Begin Time (s) | End Time (s) | Low Freq (Hz) | High Freq (Hz) |
        Species Code | Common Name | Confidence  [0.0; 1.0]
    """

    def __init__(self, raw_record: List[str]) -> None:
        self.selection = int(raw_record[0])
        self.view = raw_record[1].strip()
        self.channel = int(raw_record[2])
        begin_time = float(raw_record[3])
        end_time = float(raw_record[4])
        self.low_freq = int(raw_record[5])
        self.high_freq = int(raw_record[6])
        species_code = raw_record[7].strip()
        common_name = raw_record[8].strip()
        confidence = float(raw_record[9])
        super().__init__(begin_time, end_time, confidence, species_code, common_name)

    # tab separated
    # Selection	View	    Channel	Begin   Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Species Code	Common Name	    Confidence
    # 1	        Spectrogram 1	    1	    1380.0	    1383.0	        150	            12000	        rocpta1	        Rock Ptarmigan	0.4292
    ###############
    # 0 Selection ;
    # 1 View ;
    # 2 Channel;
    # 3 Begin Time (s);
    # 4 End Time (s) ;
    # 5 Low Freq (Hz);
    # 6 High Freq (Hz);
    # 7 Species Code;
    # 8 Common Name ;
    # 9 Confidence ;

    @staticmethod
    def parse_file(filename: str) -> List['BnCsvResultRecord']:
        """
        Reads a single BirdNet-analyzer result file into a list of records
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
            if curr == 1:
                labels = data
                continue
            current = BnCsvResultRecord(data)
            records.append(current)
        return records
