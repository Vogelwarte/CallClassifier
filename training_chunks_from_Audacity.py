import argparse
import os
import sys
from datetime import datetime
from math import ceil, floor
from pathlib import Path
from typing import List, Dict

import librosa

from BNResultTools.AudacityLabel import AudacityLabel
from BNResultTools.SynonymeChecker import SynonymeChecker
from BNResultTools.BirdNETSelectionRecord_David import BirdNETSelectionRecord_David
from BNResultTools.RelativeTimeSegment import RelativeTimeSegment, LabeledRelativeTimeSegment
from BNResultTools.Table1SelectionRecord import Table1SelectionRecord
from BNResultTools.files_dirs_tools import list_of_files, dictionary_by_bare_name


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.description = f'Extracts audio chunks from long wav file based on Audacity label file. The result consists ' \
                         f'of multiple audio file, each 3s long \'chunk\', and a one-hot_labels.csv describing ' \
                         f'presence or absence of the annotated call in all the chunks'
    parser.add_argument("-i", "--input_dir",
                        help="Path to the file or folder of the manual annotation in the *.txt (Audacity) format",
                        default=".")
    parser.add_argument("-a", "--audio-root-dir",
                        help="Path to the root directory of the audio files. Default=current working dir.", default=".")
    parser.add_argument("-o", "--output-directory",
                        help="Path to the output directory. If doesn't exist, it will be created.",
                        default="chunks_from_labels")
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


def create_output_dir_structure(results_folder: Path) -> (Path, Path):
    dt: datetime = datetime.utcnow()
    prefix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
    output_type: str = "chunks"
    out_dir = results_folder / Path(f"{prefix}_{output_type}")
    os.mkdir(out_dir)
    if not out_dir.is_dir():
        raise NotADirectoryError("Cannot create output directory " + str(out_dir))
    report_file: Path = out_dir / Path(f'one-hot_labels.csv')
    return out_dir, report_file


def generate_chunk_coverage(ts: LabeledRelativeTimeSegment, chunk_length: float, overlap: float) -> List[LabeledRelativeTimeSegment]:
    if ts.duration()<chunk_length:
        return [ts]
    if ts.duration() - chunk_length < 1.0:
        return [LabeledRelativeTimeSegment(ts.middle() - chunk_length/2.0, ts.middle()+chunk_length/2.0, ts.label)]
    step: float = chunk_length - overlap
    n_steps: int = ceil((ts.duration() - chunk_length) / step)
    real_step: float = (ts.duration() - chunk_length) / n_steps
    results_list: List[LabeledRelativeTimeSegment] = []
    first_ts: LabeledRelativeTimeSegment = LabeledRelativeTimeSegment(ts.begin_time, ts.begin_time+chunk_length, ts.label)
    for i in range(0, n_steps+1, 1):
        results_list.append( LabeledRelativeTimeSegment.from_rts(first_ts >> (i * real_step),ts.label))
    return results_list

def get_annotation_labels(fp_list:List[Path]) -> List[str]:
    labels: Dict[str,int] = {}
    for fp in fp_list:
        try:
            chunk_definitions: List[AudacityLabel] = AudacityLabel.parse_file(fp)
            for cd in chunk_definitions:
                labels[cd.label.strip().lower()] = 1
        except:
            pass
    return labels.keys()


def do_the_butchery(args):
    chunk_length: int = 3.0
    input_extension = ".txt" #".Table.1.selections.txt"  # ".wav.csv"
    chunk_def_files = list_of_files(args.input_dir, input_extension)
    af_extension = ".flac" #, ".wav"
    audio_files = dictionary_by_bare_name(list_of_files(args.audio_root_dir, af_extension), af_extension)
    # print(str(audio_files))
    odir: Path = Path(args.output_directory)
    if not odir.exists():
        os.mkdir(odir)
    exe_output_dir, report_file = create_output_dir_structure(odir)
    with open(report_file, "w") as rf:
        df_processed = 0
        print("Extracting audio fragments report file. \n")
        print(f"Input file/folder: {args.input_dir}")
        print(f"Audio root director: {args.audio_root_dir}\n")
        print(f"\nFound {len(chunk_def_files)} inpupt files.\n")
        classes: List[str] = ["_nothing"]
        classes.extend(get_annotation_labels(chunk_def_files))
        print(f'filename, ' + ','.join(classes), file=rf)
        for cdf in chunk_def_files:
            try:
                chunk_definitions: List[AudacityLabel] = AudacityLabel.parse_file(cdf)
                c_count = 0
                not_annot = 0
                name_stem: str = str(cdf.name)[0:-len(input_extension)-4]  # remove the extension and the _AGC or _SDI
                audio_file = audio_files.get(name_stem, None)
                if audio_file is None:
                    raise FileNotFoundError(name_stem)
                duration = librosa.get_duration(filename=audio_file)
                if duration < chunk_length:
                    raise RuntimeError(f'audio file {name_stem} is too short ({duration}s) ')
                not_annotated: List[LabeledRelativeTimeSegment] = []
                for i in range(0, floor(duration/chunk_length )):
                    not_annotated.append( LabeledRelativeTimeSegment( i * chunk_length,(i+1)*chunk_length, classes[0]) )

                annotated: List[LabeledRelativeTimeSegment] = []
                for cd in chunk_definitions:
                    for nt_label in not_annotated:
                        if nt_label.overlaping_time(cd) > 0 :
                            not_annotated.remove(nt_label)
                    annotated.extend(generate_chunk_coverage(ts = LabeledRelativeTimeSegment.from_rts(cd.extended(1), cd.label), chunk_length=chunk_length, overlap=1.5) )

                for chunk in annotated + not_annotated:
                    success, name = createChunks(chunk, audio_file, name_stem, exe_output_dir, chunk.label, args.resample)
                    one_hot = ['0'] * len(classes)
                    one_hot[classes.index(chunk.label)] = '1'
                    if success:
                        print(f'{name.name},' + ','.join(one_hot) ,file=rf)
                        c_count += 1
                #print(f"{str(cdf.name)}: {c_count} of {len(chunk_definitions)} chunks created, {not_annot} ignored as not '{annot_name}'", file=rf)
            except Exception as ex:
                print(f"Error while processing {str(cdf.name)}: {ex}", file=sys.stderr)

            finally:
                df_processed += 1

        print(f"\n End of processing, {df_processed} of {len(chunk_def_files)} input files processed")

# sox : offset + duration   normalised to 3s, new offset + new duration
# flac, --tag=offest_to_call

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmd_args = parse_commandline()
    print(cmd_args)
    do_the_butchery(cmd_args)
