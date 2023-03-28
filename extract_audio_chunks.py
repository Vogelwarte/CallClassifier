import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from BNResultTools.BirdNETSelectionRecord_David import BirdNETSelectionRecord_David
from BNResultTools.RelativeTimeSegment import RelativeTimeSegment
from BNResultTools.Table1SelectionRecord import Table1SelectionRecord
from BNResultTools.files_dirs_tools import list_of_files, dictionary_by_bare_name


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.description = f'Extracts audio chunks from long wav fille based on BirdNET (Raven-ish) result files and ' \
                         f'resamples it to 32kHz '
    parser.add_argument("-i", "--input",
                        help="Path to the file or folder of the manual annotation in the *.Table.1.selections.txt format",
                        default=".")
    parser.add_argument("-r", "--audio-root-dir",
                        help="Path to the root directory of the audio files. Default=current working dir.", default=".")
    parser.add_argument("-o", "--output-directory",
                        help="Path to the output directory. If doesn't exist, it will be created.", default="./chunks")
    parser.add_argument("-a", "--annotation",
                        help="Uses only these lines from the input file(s) that contain this string", default="")
    return parser.parse_args()


def createChunks(cd: Table1SelectionRecord, audio_file: Path, name_stem: str, output_dir: Path,
                 annotation: str) -> bool:
    audio_type: str = "wav"  # "ogg" # "mp3"
    a = annotation.replace("?", "_MAYBE")

    new_file_name = name_stem + "_" + cd.to_int_mmss_string().replace(":", ";") + "_(" + a + ")." + audio_type
    new_file_path: Path = output_dir / new_file_name
    ret: bool = createChunkFile(new_file_path, cd, audio_file)

    return ret, new_file_path

    # cd_p3: RelativeTimeSegment = cd.extended(3)
    # nfp_p3: Path = output_dir / Path(
    #     name_stem + "_" + cd.to_int_mmss_string().replace(":", ";") + "_(" + a + ")_3s_extended." + audio_type)
    # ret_p3: bool = createChunkFile(nfp_p3, cd_p3, audio_file)
    #
    # return ret and ret_p3, new_file_path


def createChunkFile(file_path: Path, rts: RelativeTimeSegment, audio_file: Path) -> bool:
    if file_path.exists():
        print("The file  [" + str(file_path) + "] already exists, skipping creating", file=sys.stderr)
        return False
    sys_call: str = "sox \"" + str(audio_file) + "\" \"" + str(file_path) + "\"" + " trim " + str(
        rts.begin_time) + " " + str(rts.duration()) + " " + "rate 32000"
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
    report_file: Path = Path(out_dir / f"{prefix}_summary.csv")
    return out_dir, report_file


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = parse_commandline()
    print(args)

    annot_name: str = args.annotation.strip().lower()
    input_extension = ".csv"  # ".wav.csv"
    chunk_def_files = list_of_files(args.input, input_extension)
    af_extension = ".wav"
    audio_files = dictionary_by_bare_name(list_of_files(args.audio_root_dir, af_extension), af_extension)
    # print(str(audio_files))
    odir: Path = Path(args.output_directory)
    if (not odir.exists()):
        os.mkdir(odir)
    exe_output_dir, report_file = create_output_dir_structure(odir)
    with open(report_file, "w") as rf:
        df_processed = 0
        print("Extracting audio fragments report file. \n", file=rf)
        print(f"Input file/folder: {args.input}", file=rf)
        print(f"Audio root director: {args.audio_root_dir}\n", file=rf)

        print(f"\nFound {len(chunk_def_files)} inpupt files.\n", file=rf)

        for cdf in chunk_def_files:

            chunk_definitions: List[BirdNETSelectionRecord_David] = BirdNETSelectionRecord_David.parse_file(cdf)
            c_count = 0
            not_annot = 0
            for cd in chunk_definitions:
                name_stem: str = cd.sound_files[0:-4]
                print(name_stem + " " + cd.sound_files, file=sys.stderr)
                audio_file = audio_files.get(name_stem, None)
                if audio_file is None:
                    print("cannot find audio file with name [" + name_stem + ".(wav)]", file=sys.stderr)
                    print(f"{str(cdf.name)}: cannot find the corresponding audio file", file=rf)
                    continue
                if annot_name in cd.species_code.lower():
                    success, name = createChunks(cd, audio_file, name_stem, exe_output_dir, "type_" + str(cd.type))
                    if success:
                        classes = ["0", "0", "0", "0"]
                        if cd.type <= 4:
                            classes[cd.type - 1] = "1"
                            print(str(name) + ", " + ",".join(classes), file=rf)
                        c_count += 1
                else:
                    not_annot += 1
            # print(f"{str(cdf.name)}: {c_count} of {len(chunk_definitions)} chunks created, {not_annot} ignored as not '{annot_name}'", file=rf)
            df_processed += 1

        print(f"\n End of processing, {df_processed} of {len(chunk_def_files)} input files processed", file=rf)

# sox : offset + duration   normalised to 3s, new offset + new duration
# flac, --tag=offest_to_call
