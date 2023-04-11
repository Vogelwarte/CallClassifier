import argparse
import os
import re
import sys
import traceback
from datetime import datetime
from math import ceil, floor
from pathlib import Path
from typing import List

import librosa

from BNResultTools.SynonymeChecker import SynonymeChecker
from BNResultTools.BirdNETSelectionRecord_David import BirdNETSelectionRecord_David
from BNResultTools.RelativeTimeSegment import RelativeTimeSegment
from BNResultTools.Table1SelectionRecord import Table1SelectionRecord
from BNResultTools.files_dirs_tools import list_of_files, dictionary_by_bare_name


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.description = f'Generates threshold vs precision of the validated result in the Raven format'
    parser.add_argument("-i", "--input_dir",
                        help="Path to the file or folder of the validated results in the *.Table.1.selections.txt (Raven) format",
                        default=".")
    parser.add_argument("-o", "--output-directory",
                        help="Path to the output directory. If doesn't exist, it will be created.",
                        default="chunks_rock_ptarmigan_24kHz")
    return parser.parse_args()



def create_output_dir_structure(results_folder: Path) -> (Path, Path, Path):
    os.makedirs(results_folder, exist_ok=True)
    dt: datetime = datetime.utcnow()
    suffix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
    output_type: str = "validation_result_RockPtarmigan"
    out_dir = results_folder / Path(f"{output_type}_{suffix}")
    os.mkdir(out_dir)
    if not out_dir.is_dir():
        raise NotADirectoryError("Cannot create output directory " + str(out_dir))
    log_file: Path = Path(out_dir / f'processing_log.txt')
    total_file: Path = Path(out_dir / f'total.csv')
    threshold_file: Path = Path(out_dir / f'confidence_threshold.csv')

    return out_dir, log_file, total_file, threshold_file




def to_hh_mm_ss(seconds: float) -> str:
    """
    :param seconds:
    :return: the string representation in format HHhMMmSSs with leading zeros
    """

    h:int = floor(seconds / (60*60))
    m:int = floor((seconds % (60*60)) / 60)
    s:float = seconds % 60

    ret_str: str = f'{h}h {m}min {s:.1f}s'
    return ret_str



class RPValidationAnnotation():
    def __init__(self, raw_input: str):
        self.input = raw_input
        self.normalised = re.sub(r"\s+", "", self.input.strip().lower(), flags=re.UNICODE)
        p = re.compile('(TP|FP)(RockPtarmigan)(\d\.\d{2})-(\d\.\d{2})', re.IGNORECASE)
        m = p.match(self.normalised)
        if not m:
            raise Exception(f'Invalid annotation syntax: [{self.input}]')
        self.status = m.group(1)
        self.annotation = m.group(2)
        self.min_score = float(m.group(3))
        self.max_score = float(m.group(4))




def do_the_butchery(args):
    exe_output_dir, log_fp, total_fp,  threshold_fp = create_output_dir_structure(args.output_directory)
    with open(log_fp, "w") as log:
        print(f'Commandline arguments: {args}', file=log)
        input_extension = ".selection.table.txt"  # ".wav.csv"
        chunk_def_files = list_of_files(args.input_dir, input_extension)
        with open(total_fp, "w") as total:
            print(f'Analysing the validated result set. Start of processing: {datetime.now()} \n', file=log)
            print(f'Input file/folder: {args.input_dir}', file=log)
            print(f'Found {len(chunk_def_files)} inpupt files.', file=log)
            all_records = []
            files_processed: int = 0
            print(f'filename, begin_time, end_time, min_score, max_score, status, annotation', file=total)
            for cdf in chunk_def_files:
                print(f'file {files_processed+1}. [{cdf.name}]', file=log)
                chunk_definitions: List[Table1SelectionRecord] = Table1SelectionRecord.parse_file(cdf)
                for cd in chunk_definitions:
                    try:
                        rpa = RPValidationAnnotation(cd.annotation)
                        print(f'{cdf.name}, {cd.begin_time}, {cd.end_time}, {rpa.min_score}, {rpa.max_score}, {rpa.status}, {rpa.annotation} ', file=total)
                        all_records.append((cdf.name, cd, rpa))
                    except Exception as ex:
                        print(f"Exception while processing {str(cdf.name)}: {ex}", file=log)
                    except:
                        print(f"Error while processing {str(cdf.name)}", file=log)
                    finally:
                        pass
                files_processed += 1
        print(f'\n')
        print(f'End of processing, {datetime.now()}, processed {files_processed} of {len(chunk_def_files)} input annotation files', file=log)
        with open(threshold_fp, "w") as threshold_file:
            print('"confidence threshold[%]", "validated items", '
                  '"Correct Detections", "CD-ratio/accuracy", "CD precision"'
                  ' "Not Detected", "ND-ratio", "False Positives", "FP-ratio" ',
                  file=threshold_file)
            for i in range(0,100):
                dval: float = float(i)/100.0
                tps: int = 0
                fps: int = 0
                nds: int = 0
                padding: int = 0
                for (name, cd, rpa) in all_records:
                    if rpa.max_score >= dval:
                        if rpa.status == 'tp':
                            tps += 1
                        else:
                            fps += 1
                    else:
                        if rpa.status == 'tp':
                            nds += 1
                        else:
                            padding += 1
                all: int = tps + fps + nds + padding
                print(f'{i}, {all}, '
                      f'{tps}, {float(tps)/max(1.0,float(tps+nds)):.2f}, {float(tps)/max(1.0,float(tps+fps)):.2f}, '
                      f'{nds}, {float(nds)/max(1.0,float(tps+nds)):.2f}, '
                      f'{fps}, {float(fps)/max(1.0,float(tps+fps)):.2f}  ',
                      file=threshold_file)



# sox : offset + duration   normalised to 3s, new offset + new duration
# flac, --tag=offest_to_call

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmd_args = parse_commandline()
    do_the_butchery(cmd_args)
