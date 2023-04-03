import argparse
import os
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime
from math import ceil

import pandas as pd

from contextlib import contextmanager
from pathlib import Path
from typing import List
from multiprocessing import freeze_support
from pandas import DataFrame, Series
from opensoundscape.torch.models.cnn import load_model, CNN
from opensoundscape.audio import Audio

from BNResultTools.files_dirs_tools import list_of_files_ex
from training_chunks_from_Raven import to_hh_mm_ss

usage = """
    Classifies the Rock Ptarmigan calls from the audio file

        python classify_calls_RockPtarmigan.py --i <input_file_or_dir> --o <output_dir>

    Uses a dedicated OpenSoundscape CNN model traing on the Vogelwarte data from the Swiss Alps 2021-2022.
     
    The output files are:
        <output_dir>/<audio_file>.Curlew_call_types.csv    
"""


class SingleInputDefinition:
    def __init__(self, in_files: List[Path], base_input_dir: Path, output_dir: Path):
        self.input_files: List[Path] = in_files
        self.base_input_dir = base_input_dir
        self.output_dir: Path = output_dir


def parse_command_line() -> SingleInputDefinition:
    parser = argparse.ArgumentParser(
        description='Classify Eurasian Curlew calls from audio file(s). Supported formats: wav')
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
            if_list = list_of_files_ex(in_p, extensions)
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
            os.makedirs(out_dir, exist_ok=True)

        dt: datetime = datetime.utcnow()
        suffix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
        out_dir = out_dir / f'classification_RockPtarmigan_{suffix}'
        os.makedirs(out_dir, exist_ok=True)
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
    # print(f'\nCompleteProcess: {cp}')
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
    # print(selected_df2)

    return selected_df2


def type_of_maxval(row) -> Series:
    # print(f'Argument for type_of_max: {row}\n')
    type_indices: List[str] = ["type1", "type2", "type3", "type4"]
    vect4: List[float] = row[type_indices].tolist()
    ct = vect4.index(max(vect4)) + 1
    # print(f'type{ct},{vect4[ct - 1]:.2f}')

    return pd.Series([f'type{ct}', vect4[ct - 1]])


def load_cnn_models(sample_rate: int) -> {str, CNN}:
    freeze_support()
    result_dict: {str, CNN} = {}
    # load the model
    model_names = ["resnet18_24kHz_3.0s",
                   "inception_v3_24kHz_3.0s"]
    # "resnet101_24kHz_3.0s",
    # "resnet152_24kHz_3.0s",
    for mn in model_names:
        model: CNN = load_model(f'bird_data/Rock_ptarmigan/models/{mn}/best.model')
        model.preprocessor.pipeline.load_audio.set(sample_rate=sample_rate)
        result_dict[mn] = model
    return result_dict


def get_number_of_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except:
        pass
    return os.cpu_count()


def run_call_classifier_model(audio_file: Path, model: CNN) -> DataFrame:  # , curlew_calls_df: DataFrame) -> DataFrame:
    # print("\nClassifying the call types ...\n")
    n_cpus = get_number_of_cpus()
    n_workers: int = max(1, n_cpus - 2)

    pred_start: float = time.time()
    scores_df = model.predict(num_workers=n_workers,
                              batch_size=60,
                              samples=[str(audio_file)],
                              split_files_into_clips=True,
                              overlap_fraction=2.0 / 3.0,
                              final_clip='full',
                              activation_layer='sigmoid')
    pred_end: float = time.time()
    print(f'prediction time {pred_end-pred_start:.2f}s', file=sys.stderr)
    results_df: DataFrame = scores_df
    results_df.reset_index()

    # print("Raw model results: \n")
    # print(results_df)

    return results_df


def write_BirdNET_like_result(out_fp: Path, data: DataFrame):
    idx_name = "Selection"
    headers: List[str] = ["View", "Channel", "Begin Time(s)", "End Time(s)", "Low Freq(Hz)",
                          "High Freq(Hz)", "Species Code", "Common Name", "Confidence"]
    # exemplary row (* - to be replaced with values)
    # *	Spectrogram 1	1	1954.0*	1957.0*	150
    # 12000	rocpta1	Rock Ptarmigan	0.2588*
    bn_result: DataFrame = data.apply(
        lambda row: [
            'Spectrogram 1', 1, row.start_time, row.end_time, 150,
            12000, 'rocpta1', 'Rock Ptarmigan', row.RockPtarmigan],
        axis=1,
        result_type='expand')
    if len(bn_result) == 0:
        bn_result = DataFrame(columns=headers)
    else:
        bn_result.columns = headers
    bn_result.index.rename(name=idx_name, inplace=True)
    bn_result.index += 1  # start the records with 1 (not 0)

    # print(bn_result.head())
    bn_result.to_csv(path_or_buf=out_fp, header=True, index=True, sep='\t', float_format="%.2f")


def write_Audacity_labels_result(out_fp: Path, data: DataFrame):
    report_columns = ["start_time", "end_time", "annotation"]  #
    # input columns: [start_time_, end_time, _noise, RockPtarmigan]

    audacity_labels: DataFrame = data.apply(
        lambda row: [row.start_time, row.end_time, f'RockPtarmigan:{row.RockPtarmigan:.2f}_NOT:{row._noise:.2f}'],
        axis=1,
        result_type='expand')

    # print(audacity_labels.head())
    audacity_labels.to_csv(path_or_buf=out_fp, header=False, index=False, sep='\t', float_format="%.2f")


def classify_audio_file(in_audio_fp: Path, output_dir: Path, model: CNN, model_name: str):
    print(f'Processing file: {in_audio_fp}...')

    classification_result: DataFrame = run_call_classifier_model(in_audio_fp, model)
    # expected structure: MultiIndex:[file, start_time, end_time], columns: [ _nothing, RockPtarmigan]

    # move the indices (file, start_time, end_time) to the columns
    classification_result.reset_index(inplace=True)

    # strip column names from leading and trailing whitespaces
    classification_result.rename(str.strip, inplace=True, axis='columns')

    classes: List[str] = ["_noise", "RockPtarmigan"]
    # remove all the records with score for Rock Ptarmigan below 1%
    classification_result.drop(classification_result[classification_result[classes[1]] < 0.01].index, inplace=True)

    # drop the file columne (anyway its about one audio file {in_audio_fp} only)
    classification_result.drop(columns=['file'], inplace=True, axis=1)

    file_name_no_ext: str = in_audio_fp.stem

    # classification_result DataFrame format: (numeric index), [start_time_, end_time, _noise, RockPtarmigan]
    cc_result_fp: Path = output_dir / f'{file_name_no_ext}.RpCC.selection.table.txt'
    write_BirdNET_like_result(cc_result_fp, classification_result)

    audacity_labels_fp: Path = output_dir / (f'labels-{model_name}_{file_name_no_ext}.txt')
    write_Audacity_labels_result(audacity_labels_fp, classification_result)

    print(f'End of processing, output file: {in_audio_fp}')


def do_the_stuff():
    input_def: SingleInputDefinition = parse_command_line()
    sample_rate: int = 24000  # Hz
    chunk_length: int = 3  # seconds;
    min_audio_size: float = 0.3 * 2 * sample_rate * chunk_length  # 16bit = 2Bytes, 0.3 - optimistic flac compression rate
    models = load_cnn_models(sample_rate)
    for mn in models.keys():
        mstart_proc: float = time.time()
        files_done: int = 0
        files_omitted: int = 0
        errors: int = 0
        with open(input_def.output_dir / f'{mn}_log.txt', 'w') as log_file:
            print(f'Classification by model [{mn}], '
                  f'start time: {datetime.utcnow().strftime("%Y%m%d_%H%M%SUTC")}',
                  file=log_file, flush=True)
            for in_fp in input_def.input_files:
                fstart_time: float = time.time()
                try:
                    file_size: int = in_fp.stat().st_size
                    if file_size > min_audio_size:
                        file_out_dir: Path = input_def.output_dir / Path(mn) / in_fp.relative_to(
                            input_def.base_input_dir).parent
                        os.makedirs(file_out_dir, exist_ok=True)
                        classify_audio_file(in_fp, file_out_dir, models[mn], mn)
                        files_done += 1
                        print(f'Processing time {time.time() - fstart_time:.1f}s, '
                              f'file size {file_size / 1024.0 / 1024.0:.1f}MB, '
                              f'file:[{in_fp}]', file=log_file, flush=True)
                    else:
                        files_omitted += 1
                        print(f'File size too small: '
                              f'{in_fp.stat().st_size / 1024.0:.1f}kB, '
                              f'min: {min_audio_size / 1024.0:.1f}kB, '
                              f'file:[{in_fp}]', file=log_file)
                    log_file.flush()
                except Exception as ex:
                    errors += 1
                    print(f'Exception with model {mn}, audio file {in_fp}:\n{ex}', file=log_file)
                except:
                    errors += 1
                    print(f'Unknown error with model {mn}, audio file {in_fp}', file=log_file)
            mstart_proc = time.time() - mstart_proc
            print(f'Model {mn}, files done: {files_done}, omitted: {files_omitted}, errors:{errors}, '
                  f'processing time: {to_hh_mm_ss(mstart_proc)} ({mstart_proc:.1f}s), '
                  f'speed: {mstart_proc / max(1.0, files_done):.1f} s/file',
                  file=log_file)


@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


if __name__ == '__main__':
    start: float = time.time()
    if platform.system() == 'Windows':
        with set_posix_windows():
            do_the_stuff()
    else:
        do_the_stuff()
    end: float = time.time()
    print(f'\nTotal processing time: {int(end - start)}s')
    sys.exit(0)
