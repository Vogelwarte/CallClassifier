import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Dict


class FileMetadataRecord:
    """

    """
    def __init__(self, file_path: Path, recorder_type: str, duration: float,
                 location: str, area: str, season: str, period:str, audio_metadata:str,
                 utc_tz_offset:timedelta = None):
        self.file_path = file_path
        self.recorder_type = recorder_type
        self.duration = duration
        self.location = location
        self.area = area
        self.season = season
        self.period = period
        self.audio_metadata = audio_metadata
        self.utc_tz_offset = utc_tz_offset


    def get_file_name(self)->str:
        return self.file_path.name



    @staticmethod
    def simplify_filename(name: str ) -> str:
        return name.replace("-", "").replace("_", "").upper()

    def get_simplified_file_name(self) -> str:
        return FileMetadataRecord.simplify_filename(self.get_file_name())

    def to_csv_string(self, separator:str = "; ", relative_root_path: Path = None) -> str:
        full_path = self.file_path
        if relative_root_path is not None:
            full_path = full_path.relative_to(relative_root_path)
        tz_off = '?'
        if self.utc_tz_offset is not None:
            tz_off = str(self.utc_tz_offset)
        return separator.join(
            ['"' + self.get_file_name() + '"',
             str(round(self.duration,3)),
             '"' + self.recorder_type + '"',
             '"' + self.location + '"',
             '"' + self.area + '"',
             '"' + self.season + '"',
             '"' + self.period + '"',
             '"' + tz_off + '"',
             '"' + str(full_path) + '"',
             '"' + self.audio_metadata + '"'
             ]
        )

    @staticmethod
    def csv_headers_line(separator="; ") -> str:
        return separator.join(
            ['"file name"', #0
             '"duration [s]"',  #1
             '"recorder_type"', #2
             '"location"',  #3
             '"area"',  #4
             '"season"',    #5
             '"period"',    #6
             '"UTC timezone offset"',   #7
             '"full path"', #8
             '"audio_metadata"'] #9
        )

    @staticmethod
    def parse_csv_file_to_dict(filename: str, recorder_type:str, separator:str = ";") -> Dict[str,'FileMetadataRecord']:
        """
        """
        rt = recorder_type.upper()
        records = {}
        file = open(filename, "r")
        n_lines = 0
        for line in file.readlines():
            line = line.strip()
            n_lines += 1
            data = line.split(separator)
            for i in range(0, len(data) ):
                data[i] = data[i].strip().strip('"')
            if n_lines == 1:
                labels = data
                continue
            if rt != data[2].upper():
                print("Skipping meta data of ["+data[8]+"], because of recorder type ["+data[2]+"]!=["+rt+"]", file=sys.stderr)
                continue
            am = ""
            if len(data) >= 10:
                am = data[9]
            current = FileMetadataRecord(file_path=Path(data[8]),
                                         recorder_type=data[2],
                                         duration=float(data[1]),
                                         location=data[3],
                                         area=data[4],
                                         season=data[5],
                                         period=data[6],
                                         audio_metadata = am)
            current.utc_tz_offset = data[7]
            #prev_s = len(records)
            k = current.get_simplified_file_name()
            already_in = records.get(k)
            if already_in is not None:
                print("Not unique filename found the the csv with metadata for ["+k+"]\n\t "
                      +"keeping the first candidate ["+str(already_in.file_path) +"] "
                      +" instead of the new: [" + str(current.file_path)+"]", file=sys.stderr)
                #raise Exception("Not unique filename ["+k+" : "+str(current.file_path)+"] in csv file ["+filename+"]")
            else:
                records[k] = current
                #print("Added SMMini file : [" + str(current.file_path)+"]", file=sys.stderr)
        return records
