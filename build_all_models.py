# the cnn module provides classes for training/predicting with various types of CNNs
import argparse
import os
import sys
from datetime import datetime
# other utilities and packages
# import pandas as pd
from importlib import metadata
from pathlib import Path
from typing import List, Dict

import librosa
import pandas as pd
# set up plotting
from matplotlib import pyplot as plt
from matplotlib_inline.config import InlineBackend
from opensoundscape.torch.architectures import cnn_architectures
from opensoundscape.torch.models.cnn import CNN, use_resample_loss, InceptionV3
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

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
    # parser.add_argument("-d", "--duration", help="Call duration in seconds. Default=3s", default=3.0, type=float)
    parser.add_argument("-a", "--architecture", help="Name of the architecture. If not given, all available "
                                                     "architectures will be used", type=str)
    parser.add_argument("-r", "--sample_rate", help="Sample rate to be used in the model, in Hz. Default=32000Hz",
                        default=32000, type=int)
    parser.add_argument("-m", "--multi_target",
                        help="Buld a multi-target model. If nod given, the model is single-target",
                        default=False, type=bool)

    return parser.parse_args()


def get_number_of_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except:
        pass
    return os.cpu_count()


def get_sample_rate(row) -> Series:
    sr: float = -1.0
    try:
        sr = librosa.get_samplerate(row['filename'])
    except Exception as ex:
        pass
    return pd.Series([sr])


def log_input_filelist_error(fl: DataFrame, title: str):
    print(title, file=sys.stderr)
    print(f'{len(fl)} file(s):', file=sys.stderr)
    print(fl.head(), file=sys.stderr)


def load_fileset(dir_with_files: Path, expected_sample_rate: int, log_file) -> DataFrame:
    table: DataFrame = pd.read_csv(dir_with_files / Path("one-hot_labels.csv"))
    table.filename = [str(dir_with_files / Path(f)) for f in table.filename]

    table['SR'] = table.apply(get_sample_rate, axis=1)
    log_input_filelist_error(table[table.SR == -1], 'Files not found or not audio')
    log_input_filelist_error(table.loc[((table.SR > 0) & (table.SR != expected_sample_rate))],
                             f'Sample rate ({expected_sample_rate / 1000}kHz) mismatch ')
    ct2 = table[table.SR == expected_sample_rate]
    table = ct2
    table = table.drop('SR', axis=1)

    for fp in table['filename']:
        sr: float = librosa.get_samplerate(fp)
        print(f'File [{fp}]: {sr / 1000.0:.0f}kHz', file=log_file)
        if sr != expected_sample_rate:
            print(
                f'File {fp} has {sr / 1000.0:.0f}kHz sampling rate, expected {expected_sample_rate}. Resampling not '
                f'implemented for training',
                file=log_file)
            sys.exit(1)
    print('\n')
    table = table.set_index('filename')
    return table


def prepare_training_data(data_dir: Path, expected_sample_rate: int, duration: float, output_dir: Path):
    with open(output_dir / Path(f'training_data_log'), "w") as log_file:
        version = metadata.version("opensoundscape")
        print(f'OpenSoundscape version: {version}.', file=log_file)

        training_dir: Path = data_dir / Path(f'training')
        validation_dir: Path = data_dir / Path(f'validation')
        labels: [str] = []
        if training_dir.is_dir() and validation_dir.is_dir():
            print(f'Training files:', file=log_file)
            training_df = load_fileset(training_dir, expected_sample_rate, log_file)
            print(f'Validation files', file=log_file)
            validation_df = load_fileset(validation_dir, expected_sample_rate, log_file)
            labels = training_df.columns.values.tolist()
        else:
            all_df: DataFrame = load_fileset(data_dir, expected_sample_rate, log_file)
            labels = all_df.columns.values.tolist()
            print(all_df.head())
            print(f'labels: {labels} ')

            training_df, validation_df = train_test_split(all_df, test_size=0.25, random_state=1)

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
                sample_rate_Hz: int, n_epochs: int, single_target: bool):
    model_id: str = f'{arch}_{(int)(sample_rate_Hz / 1000)}kHz_{duration}s'
    print(f'Training the model [ {model_id} ]')
    # Create model object
    classes = train_df.columns

    model: CNN = None
    if arch.lower().startswith("inception_v3"):
        model = InceptionV3(classes=classes, sample_duration=duration, single_target=single_target)
    else:
        model = CNN(architecture=arch, classes=classes, sample_duration=duration, single_target=single_target)

    model.preprocessor.pipeline.load_audio.set(sample_rate=sample_rate_Hz)
    use_resample_loss(model)

    model_out_dir: Path = output_dir / Path(f'{model_id}')
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
        batch_size=64,
        save_interval=5,  # save model every 5 epochs (the best model is always saved in addition)
        num_workers=n_workers,  # specify 4 if you have 4 CPU processes, eg; 0 means only the root process
    )
    # Save the Loss History
    save_loss_history(model_id, model_out_dir, model.loss_hist)


def build_models(output_dir: Path, train_df: DataFrame, validate_df: DataFrame, duration: float, sample_rate_Hz: int,
                 n_epochs: int, single_target: bool):
    print(f'Building models sample duration: {duration}s, sample rate: {sample_rate_Hz}Hz')
    archs: List[str] = cnn_architectures.list_architectures()
    archs = ['inception_v3', 'resnet18', 'resnet152', 'efficientnet_b0']
    for arch in archs:
        try:
            build_model(output_dir, arch, train_df, validate_df, duration, sample_rate_Hz, n_epochs, single_target)
        except Exception as ex:
            print(f'Exception occurred while training the {arch} model: {ex}')


def start_building(args):
    dt: datetime = datetime.now()
    suffix: str = dt.strftime("%Y%m%d_%H%M%SUTC")
    out_dir = Path(f'./trained_models_{suffix}')
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_duration: float = 3.0

    train_df, validate_df = prepare_training_data(args.input_data_dir, args.sample_rate, chunk_duration, out_dir)
    build_models(out_dir, train_df, validate_df, chunk_duration, args.sample_rate, args.epochs, (not args.multi_target))


def list_architectures():
    version = version = metadata.version("opensoundscape")
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
