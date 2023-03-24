import argparse
import os
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
    parser.description = f'Extracts audio chunks from long wav file based on Raven annotation. The result consists ' \
                         f'of multiple audio file, each 3s long \'chunk\', and a one-hot_labels.csv describing ' \
                         f'presence or absence of the annotated call in all the chunks'
    parser.add_argument("-i", "--input_dir",
                        help="Path to the file or folder of the manual annotation in the *.Table.1.selections.txt (Raven) format",
                        default=".")
    parser.add_argument("-a", "--audio-root-dir",
                        help="Path to the root directory of the audio files. Default=current working dir.", default=".")
    parser.add_argument("-o", "--output-directory",
                        help="Path to the output directory. If doesn't exist, it will be created.",
                        default="chunks_rock_ptarmigan_24kHz")
    # parser.add_argument("-a", "--annotation",
    #                     help="Uses only these lines from the input file(s) that contain this string", default="")
    parser.add_argument("-r", "--resample", help="Resample the chunk to the given value in Hz.", type=int,
                        default=24000)
    return parser.parse_args()


def createChunks(cd: RelativeTimeSegment, audio_file: Path, name_stem: str, output_dir: Path,
                 annotation: str, sample_rate: int) -> bool:
    audio_type: str = 'flac' #"wav"  # "ogg" # "mp3"
    a = annotation.replace("?", "_MAYBE")

    new_file_name = name_stem + "_" + cd.to_int_mmss_string().replace(":", ";") + "_(" + a + ")." + audio_type
    new_file_path: Path = output_dir / new_file_name
    ret: bool = createChunkFile(new_file_path, cd, audio_file, sample_rate)

    return ret, new_file_path

    # cd_p3: RelativeTimeSegment = cd.extended(3)
    # nfp_p3: Path = output_dir / Path(
    #     name_stem + "_" + cd.to_int_mmss_string().replace(":", ";") + "_(" + a + ")_3s_extended." + audio_type)
    # ret_p3: bool = createChunkFile(nfp_p3, cd_p3, audio_file)
    #
    # return ret and ret_p3, new_file_path


def createChunkFile(file_path: Path, rts: RelativeTimeSegment, audio_file: Path, sample_rate: int) -> bool:
    if file_path.exists():
        print("The file  [" + str(file_path) + "] already exists, skipping creating", file=sys.stderr)
        return False
    sys_call: str = "sox \"" + str(audio_file) + "\" \"" + str(file_path) + "\"" + " trim " + str(
        rts.begin_time) + " " + str(rts.duration()) + " " + f'rate {sample_rate}'
    print(sys_call);
    exit_code = os.system(sys_call)
    if exit_code != 0:
        print("Error! The command  [" + sys_call + "] returned [" + str(exit_code) + "]", file=sys.stderr)
        return False
    return True


def create_output_dir_structure(results_folder: Path) -> (Path, Path,Path):
    dt: datetime = datetime.utcnow()
    prefix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
    output_type: str = "chunks"
    out_dir = results_folder / Path(f"{prefix}_{output_type}")
    os.mkdir(out_dir)
    if not out_dir.is_dir():
        raise NotADirectoryError("Cannot create output directory " + str(out_dir))
    log_file: Path = Path(out_dir / f'processing_log.txt')
    label_file: Path = Path(out_dir / f'one-hot_labels.csv')

    return out_dir, log_file, label_file


def generate_chunk_coverage(ts: RelativeTimeSegment, chunk_length: float, min_overlap: float) -> List[RelativeTimeSegment]:
    if ts.duration() <= chunk_length:
        return [ts]
    if ts.duration() - chunk_length < 1.0:
        return [RelativeTimeSegment(ts.middle() - chunk_length/2.0, ts.middle()+chunk_length/2.0)]
    step: float = chunk_length - min_overlap

    n_steps: int = ceil((ts.duration() - chunk_length) / step)
    real_step: float = (ts.duration() - chunk_length) / n_steps
    results_list: List[RelativeTimeSegment] = []
    first_ts: RelativeTimeSegment = RelativeTimeSegment(ts.begin_time, ts.begin_time+chunk_length)
    for i in range(0, n_steps+1, 1):
        results_list.append(first_ts >> (i * real_step))
    return results_list


def do_the_butchery(args):
    chunk_length: int = 3.0
    rp_name_checker: SynonymeChecker = SynonymeChecker.parse_file("rock_ptarmigan_synonym_regexps.txt")
    input_extension = ".Table.1.selections.txt"  # ".wav.csv"
    chunk_def_files = list_of_files(args.input_dir, input_extension)
    af_extension = ".wav"
    audio_files = dictionary_by_bare_name(list_of_files(args.audio_root_dir, af_extension), af_extension)
    # print(str(audio_files))
    odir: Path = Path(args.output_directory)
    if not odir.exists():
        os.mkdir(odir)
    exe_output_dir, log_fp, labels_fp = create_output_dir_structure(odir)
    with open(log_fp, "w") as log, open(labels_fp, "w") as labels:
        df_processed = 0
        print(f'Extracting audio fragments report file. Start of processing: {datetime.now()} \n', file=log)
        print(f'Input file/folder: {args.input_dir}', file=log)
        print(f'Audio root director: {args.audio_root_dir}\n', file=log)
        print(f'\nFound {len(chunk_def_files)} inpupt files.\n', file=log)

        str_present: str = 'RockPtarmigan'
        str_absent: str = '_noise'
        print(f'filename, {str_absent}, {str_present}', file=labels)
        for cdf in chunk_def_files:
            try:
                chunk_definitions: List[Table1SelectionRecord] = Table1SelectionRecord.parse_file(cdf)
                c_count = 0
                not_annot = 0
                name_stem: str = str(cdf.name)[0:-len(input_extension)]
                audio_file = audio_files.get(name_stem, None)
                if audio_file is None:
                    print(f'Cannot fing audiofile for {cdf.name}, skipping this annotation')
                    continue
                duration = librosa.get_duration(filename=audio_file)
                if duration < chunk_length:
                    raise RuntimeError(f'audio file {name_stem} is too short ({duration}s) ')
                absents: List[RelativeTimeSegment] = []
                for i in range(0, floor(duration/chunk_length )):
                    absents.append( RelativeTimeSegment( i * chunk_length,(i+1)*chunk_length) )
                presents: List[RelativeTimeSegment] = []
                for cd in chunk_definitions:
                    if rp_name_checker.is_synonyme(cd.annotation.lower()):
                        for a in absents:
                            if a.overlaping_time(cd) > 0:
                                absents.remove(a)
                        if cd.duration() < max(1, chunk_length - 2):
                            continue
                        presents.extend(generate_chunk_coverage(ts = cd.extended(1.0), chunk_length=chunk_length, min_overlap=1.2) )

                for chunk in presents:
                    success, name = createChunks(chunk, audio_file, name_stem, exe_output_dir, str_present, args.resample)
                    if success:
                        print(f'{str(name)} , 0, 1', file=labels)
                        c_count += 1
                n_absent = ceil(duration / 60)
                every_n: int = ceil( len(absents) / n_absent)
                i:int = 0
                for chunk in absents:
                    i+=1
                    if i % every_n != 0:
                        continue
                    success, name = createChunks(chunk, audio_file, name_stem, exe_output_dir, str_absent, args.resample)
                    if success:
                        print(f'{str(name)} , 1, 0', file=labels)
                        not_annot += 1
            #print(f"{str(cdf.name)}: {c_count} of {len(chunk_definitions)} chunks created, {not_annot} ignored as not '{annot_name}'", file=rf)

            except Exception as ex:
                print(f"Error while processing {str(cdf.name)}: {ex}", file=log)
                traceback.print_stack(file=sys.stderr)
            finally:
               df_processed += 1

        print(f"\n End of processing, {df_processed} of {len(chunk_def_files)} input files processed, {datetime.now()} ", file=log)


# sox : offset + duration   normalised to 3s, new offset + new duration
# flac, --tag=offest_to_call

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmd_args = parse_commandline()
    print(cmd_args)
    do_the_butchery(cmd_args)
