import argparse
import os
import subprocess
import sys
import time
import pandas as pd

from pathlib import Path
from typing import List
from multiprocessing import freeze_support
from pandas import DataFrame, Series
from opensoundscape.torch.models.cnn import load_model, CNN
from opensoundscape.audio import Audio

from BNResultTools.files_dirs_tools import list_of_files_ex

usage = """
    Classifies the European Ceurlew call from the audio file

        python classify_calls.py --i <input_file_or_dir> --o <output_dir>

    BirdNET v2.2 is used to identify the curlew vocalisations. Then a dedicated CNN model is used 
    to distinguish the different types of the calls.

    The output files are:
        <output_dir>/<audio_file>.BirdNET_labels.txt    (Audacity format) 
        <output_dir>/<audio_file>.Curlew_call_types.csv    
"""


class SingleInputDefinition:
    def __init__(self, in_files: List[Path], base_input_dir: Path,  output_dir: Path):
        self.input_files: List[Path] = in_files
        self.base_input_dir = base_input_dir
        self.output_dir: Path = output_dir


def parse_command_line() -> SingleInputDefinition:
    parser = argparse.ArgumentParser(description='Classify Eurasian Curlew calls from audio file(s). Supported formats: wav')
    parser.add_argument('--i',
                        help='Path to input file or folder. If this is a file, --o needs to be a file too.')
    parser.add_argument('--o',
                        help="Path to output folder. It will be created id it doesn't exists")
    args = parser.parse_args()
    extensions: List[str] = ["wav", "flac"]
    try:
        if_list: List[Path] = []
        in_p = Path(args.i)
        base_dir: Path = in_p.parent
        if in_p.is_dir():
            if_list = list_of_files_ex(in_p, extensions )
            base_dir = in_p
        else:
            if in_p.is_file():
                for ext in extensions:
                    if in_p.name.lower().endswith(ext):
                        if_list = [in_p]
                        break
                if len(if_list) == 0:
                    raise RuntimeError(f'{in_p}: invalid filetype. Only {extensions} files are supported')
            else:
                raise Exception(f'{in_p} is neither file nor directory')
        out_dir = Path(args.o)
        if out_dir.exists():
            if not out_dir.is_dir():
                raise NotADirectoryError(f'{out_dir} cannot be an output folder')
        else:
            os.mkdir(out_dir)
        return SingleInputDefinition(if_list, base_dir, out_dir)
    except Exception as ex:
        print(ex, file=sys.stderr)
        print(usage, file=sys.stdout)
        sys.exit(1)


def get_birdnet_results(infile: Path, output_dir: Path, common_name: str, confidence: float) -> DataFrame:
    print(f"Running BirdNET-Analyzer V2.2 on {infile}")
    bn_dir: Path = Path("./BirdNET-Analyzer_V2.2")
    bn_analyzer: str = r'analyze.py'

    bnOutFP: Path = output_dir / (str(infile.name) + ".BirdNET_labels.txt")

    wd: Path = Path(sys.argv[0]).parent.absolute()

    cp = CompleteProcess = subprocess.run(
        ['python',
         str(bn_dir / bn_analyzer),
         "--i", str(infile),
         "--o", str(bnOutFP),
         "--rtype", "audacity",
         "--slist", str(Path("..") / "binary_train")
         ])
    #print(f'\nCompleteProcess: {cp}')
    if cp.returncode != 0:
        raise RuntimeError(f'BirdNET analysis failed: {cp}')
    bn_out_df: DataFrame = pd.read_table(bnOutFP, header=None, names=["start_time", "end_time", "name", "confidence"])

    # print("BirdNet results:\n")
    # print(bn_out_df)

    selected_df = bn_out_df[bn_out_df["name"].str.endswith(common_name)]
    selected_df2 = selected_df[selected_df["confidence"] >= confidence]
    selected_df2["name"] = common_name
    selected_df2.reset_index(drop=True, inplace=True)
    print(f'BirdNET found {len(selected_df2)} {common_name} calls')
    #print(selected_df2)

    return selected_df2


def type_of_maxval(row) -> Series:
    # print(f'Argument for type_of_max: {row}\n')
    type_indices: List[str] = ["type1", "type2", "type3", "type4"]
    vect4: List[float] = row[type_indices].tolist()
    ct = vect4.index(max(vect4)) + 1
    # print(f'type{ct},{vect4[ct - 1]:.2f}')

    return pd.Series([f'type{ct}', vect4[ct - 1]])


def run_call_classifier_model(audio_file: Path, curlew_calls_df: DataFrame) -> DataFrame:
    freeze_support()
    # load the model
    model: CNN = load_model('binary_train/best.model')

    model.preprocessor.pipeline.load_audio.set(sample_rate=32000)
    # print("\nClassifying the call types ...\n")
    scores_df, preds_df, labels_df = model.predict(num_workers=8,
                                                   samples=[str(audio_file)],
                                                   split_files_into_clips=True,
                                                   final_clip='full',
                                                   binary_preds=None,
                                                   activation_layer='sigmoid')
    results_df: DataFrame = scores_df
    results_df.reset_index()
    # report_file: Path = Path("raw_types.csv")
    # results_df.to_csv(report_file)
    # results_df = pd.read_csv(report_file)

    #print("Raw model results: \n")
    #print(results_df)
    joint_result_df: DataFrame = pd.merge(curlew_calls_df, results_df, how='inner', on=["start_time", "end_time"])
    joint_result_df[["call_type", "type_confidence"]] = joint_result_df.apply(type_of_maxval, axis=1)

    #print("Joint results:\n")
    #print(joint_result_df)

    return joint_result_df


def classify_audio_file(in_audio_fp: Path, output_dir: Path):
    print(f'Processing file: {in_audio_fp}...')

    # run birdnet on the file
    # get the result of BirdNet (BN) and select Eurasian Curlew | "eurcur, any confidence
    curlew_calls_df: DataFrame = get_birdnet_results(in_audio_fp, output_dir, "Eurasian Curlew", 0.0)

    # Pshemek: openesoundscape documentation has an error:
    # CNN.predict doesn't handle the MultiIndex input DataFrame [file, start_time, end_time]
    # run the call_classifier model on whole file and pick only the chunks recognized by BirdNET
    report_columns = ["start_time", "end_time", "name", "confidence", "call_type", "type_confidence"]
    if len(curlew_calls_df) > 0:
        call_types_df: DataFrame = run_call_classifier_model(in_audio_fp, curlew_calls_df)
    else:
        call_types_df: DataFrame = DataFrame([], columns=report_columns)
        #print(f'[{in_audio_fp}+"]: no Eurasian Curlew calls found"')

    # output the results
    joint_report_file: Path = output_dir / (in_audio_fp.name + ".Curlew_call_types.csv")
    call_types_df.to_csv(joint_report_file,
                         index=False,
                         columns=report_columns)
    print(f'End of processing, output file: {joint_report_file}')

def do_the_stuff():
    input_def: SingleInputDefinition = parse_command_line()
    for in_fp in input_def.input_files:
        try:
            out_dir: Path = input_def.output_dir / in_fp.relative_to(input_def.base_input_dir).parent
            classify_audio_file(in_fp, out_dir)
        except Exception as ex:
            print(f'Error with file {in_fp}:\n{ex}', file=sys.stderr)


if __name__ == '__main__':
    start: float = time.time()
    do_the_stuff()
    end: float = time.time()
    print(f'\nTotal processing time: {int(end - start)}s')
    sys.exit(0)
