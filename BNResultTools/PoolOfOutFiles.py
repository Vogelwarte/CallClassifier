from pathlib import Path
from typing import TextIO, Dict, List


class PoolOfOutFiles:

    def __init__(self, dir: Path, rec_file_headers_line: str, voc_file_hearers_line: str):
        self.dir = dir;
        self.pool: Dict[str, (TextIO, TextIO)] = {}

        self.all_recordings_file = None
        self.all_vocalisations_file = None

        self.created_files: List[Path] = []
        self.rec_headers = rec_file_headers_line
        self.voc_headers = voc_file_hearers_line

    def get_out_files(self, label: str) -> (TextIO, TextIO):
        if label not in self.pool:
            rof_path = Path(self.dir / Path(label + ".recordings.csv"))
            rec_out_file = open(rof_path, "w")
            print(self.rec_headers, file=rec_out_file)

            self.created_files.append(rof_path)
            vof_path = Path(self.dir / Path(label + ".vocalisations.csv"))
            voc_out_file = open(vof_path, "w")
            print(self.voc_headers, file=voc_out_file)

            self.created_files.append(vof_path)
            self.pool[label] = (rec_out_file, voc_out_file)
        return self.pool[label]

    def get_all_recording_file(self) -> TextIO:
        if self.all_recordings_file == None:
            all_rec_of_path = Path(self.dir / "all_recordings.csv")
            self.all_recordings_file = open(all_rec_of_path, "w")
            print(self.rec_headers, file=self.all_recordings_file)
        return self.all_recordings_file

    def get_all_vocalisations_file(self) -> TextIO:
        if self.all_vocalisations_file == None:
            all_voc_of_path = Path(self.dir / "all_vocalisations.csv")
            self.all_vocalisations_file = open(all_voc_of_path,"w")
            print(self.voc_headers , file= self.all_vocalisations_file)
        return self.all_vocalisations_file