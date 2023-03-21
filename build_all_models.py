# the cnn module provides classes for training/predicting with various types of CNNs
import argparse
import os
import sys
from datetime import datetime
# other utilities and packages
# import pandas as pd
from pathlib import Path
from typing import List, Dict

import librosa
import pandas as pd
import pkg_resources
# set up plotting
from matplotlib import pyplot as plt
from matplotlib_inline.config import InlineBackend
from opensoundscape.torch.architectures import cnn_architectures
from opensoundscape.torch.models.cnn import CNN, use_resample_loss
from pandas import DataFrame, Series

plt.rcParams['figure.figsize'] = [15, 5]  # for large visuals
InlineBackend.figure_format = 'retina'


def parse_commandline_args():
    parser = argparse.ArgumentParser(prog='build_all_models',
                                     description='Builds CNN models for all available architectures in OpenSoundscape')
    parser.add_argument("-l", "--list", help="List the available architectures and exit", action='store_true')
    parser.add_argument("-i", "--input_data_dir",
                        help=f'Path to the folder with the audio and the one-hot labels file "one-hot_labels.csv" ',
                        default=".")
    parser.add_argument("-e", "--epochs", help=f'Number of epochs in the training', type=int, default=100)
    parser.add_argument("-d", "--duration", help="Call duration in seconds. Default=3s", default=3.0, type=float)
    parser.add_argument("-r", "--sample_rate", help="Sample rate to be used in the model, in Hz. Default=16000Hz",
                        default=32000, type=int)
    return parser.parse_args()


def get_number_of_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except:
        pass
    return os.cpu_count()


def get_sample_rate(row) -> Series:
    sr: int = -1
    try:
        sr = librosa.get_samplerate(row['filename'])
    except Exception as ex:
        pass
    return pd.Series([sr])


def load_fileset(dir_with_files: Path, expected_sample_rate: int, log_file) -> DataFrame:
    table: DataFrame = pd.read_csv(dir_with_files / Path("one-hot_labels.csv"))
    table.filename = [str(dir_with_files / Path(f)) for f in table.filename]

    table['SR'] = table.apply(get_sample_rate, axis=1)
    ct2 = table[table.SR == expected_sample_rate]
    table = ct2
    table = table.drop('SR', axis=1)

    for fp in table['filename']:
        sr: int = librosa.get_samplerate(fp)
        print(f'File [{fp}]: {sr / 1000}kHz', file=log_file)
        if sr != expected_sample_rate:
            print(
                f'File {fp} has {sr / 1000}kHz sampling rate, expected {expected_sample_rate}. Resampling not '
                f'implemented for training',
                file=log_file)
            sys.exit(1)
    print('\n')
    table = table.set_index('filename')
    return table


def prepare_training_data(data_dir: Path, expected_sample_rate: int, duration: int, output_dir: Path):
    with open(output_dir / Path(f'training_data_log'), "w") as log_file:
        version = pkg_resources.get_distribution("opensoundscape").version
        print(f'OpenSoundscape version: {version}.', file=log_file)
        print(f'Training files:', file=log_file)
        training_df = load_fileset(data_dir / Path(f'training'), expected_sample_rate, log_file)
        print(f'Validation files', file=log_file)
        validation_df = load_fileset(data_dir / Path(f'validation'), expected_sample_rate, log_file)

        labels = training_df.columns.values.tolist()
        print(f'Training set classes: {str(labels)}', file=log_file)
        print(
            f'Training set size: {len(training_df.index)}, x {duration}s = {(len(training_df.index) * duration) / 3600.0:.2f}h',
            file=log_file)
        print(
            f'Validation set size: {len(validation_df.index)}, x {duration}s = {(len(validation_df.index) * duration) / 3600.0:.2f}h',
            file=log_file)
        tl = len(training_df.index)
        vl = len(validation_df.index)
        print("Sample count per classes:", file=log_file)
        for c in labels:
            ts = training_df[c].sum()
            vs = validation_df[c].sum()
            print(
                f'  {c}: training {ts} ({(100.0 * ts) / tl:.0f}%), validation {vs} ({(100.0 * vs / vl):.0f}%), '
                f'#validation/#training {(100.0 * vs / (ts + vs)):.0f}%)',
                file=log_file)
        return training_df, validation_df


def save_loss_history(model_id: str, out_dir: Path, loss_history: Dict[int, float]):
    train_graph_fn: str = str(out_dir / Path(f'{model_id}_loss_history.png'))
    loss_history_fn: Path = out_dir / Path(f'{model_id}_loss_history.csv')

    plt.scatter(loss_history.keys(), loss_history.values(), label=model_id)
    plt.title(f'Loss history')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(train_graph_fn)
    loss_history_df: DataFrame = DataFrame.from_dict(data=loss_history, orient='index', columns=['loss'])
    loss_history_df.index.name = 'epoch'
    loss_history_df.to_csv(loss_history_fn, sep=';')


def build_model(output_dir: Path, arch: str, train_df: DataFrame, validate_df: DataFrame, duration: float,
                sample_rate_Hz: int, n_epochs: int):
    model_id: str = f'{arch}_{(int)(sample_rate_Hz / 1000)}kHz_{duration}s'
    print(f'Training the model [ {model_id} ]')
    # Create model object
    classes = train_df.columns

    model = CNN(arch, classes, duration, single_target=False)
    model.preprocessor.pipeline.load_audio.set(sample_rate=sample_rate_Hz)
    use_resample_loss(model)

    model_out_dir: Path = output_dir / Path(f'model_{model_id}')
    model_out_dir.mkdir(parents=True, exist_ok=True)
    print("model.single_target:", model.single_target)
    # Logging the Model preformance
    model.verbose = 0  # don't print anything to the screen during training
    model.logging_level = 3  # request lots of logged content
    model.log_file = str(model_out_dir / Path(f'training_log.txt'))  # specify a file to log output to,

    n_cpus = get_number_of_cpus()
    n_workers: int = max(1, n_cpus - 2)
    print(f'Setting up {n_workers} workers of {n_cpus} CPUs')

    # Train the Model
    model.train(
        train_df=train_df,
        validation_df=validate_df,
        save_path=model_out_dir,  # where to save the trained model
        epochs=n_epochs,
        batch_size=20,
        save_interval=5,  # save model every 5 epochs (the best model is always saved in addition)
        num_workers=n_workers,  # specify 4 if you have 4 CPU processes, eg; 0 means only the root process
    )
    # Save the Loss History
    save_loss_history(model_id, model_out_dir, model.loss_hist)


def build_models(output_dir: Path, train_df: DataFrame, validate_df: DataFrame, duration: float, sample_rate_Hz: int,
                 n_epochs: int):
    print(f'Building models sample duration: {duration}s, sample rate: {sample_rate_Hz}Hz')
    archs: List[str] = cnn_architectures.list_architectures()
    for arch in archs:
        try:
            if arch.lower().startswith("inception_v3"):
                raise NotImplementedError("Omitted InceptionV3 arch - special training data not implemented")
            build_model(output_dir, arch, train_df, validate_df, duration, sample_rate_Hz, n_epochs)
        except Exception as ex:
            print(f'Exception occurred while training the {arch} model: {ex}')


def start_building(args):
    dt: datetime = datetime.now()
    suffix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
    out_dir = Path(f'./trained_models_{suffix}')
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, validate_df = prepare_training_data(args.input_data_dir, args.sample_rate, args.duration, out_dir)
    build_models(out_dir, train_df, validate_df, args.duration, args.sample_rate, args.epochs)


def list_architectures():
    version = pkg_resources.get_distribution("opensoundscape").version
    #    model = CNN('resnet18', ["t1", "t2", "t3"], 3.0, single_target=False)
    print(f'OpenSoundscape version: {version}.\nAvailable CNN architectures:')
    archs: List[str] = cnn_architectures.list_architectures()
    for a in archs:
        print(a)


if __name__ == '__main__':
    cmd_args = parse_commandline_args()
    if cmd_args.list:
        list_architectures()
        sys.exit(0)
    start_building(cmd_args)
