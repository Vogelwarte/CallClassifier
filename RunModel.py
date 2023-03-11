import sys
from glob import glob
from multiprocessing import freeze_support
from pathlib import Path

from opensoundscape.torch.models.cnn import load_model, CNN
# Setup Data for Model
from pandas import DataFrame

if __name__ == '__main__':
    freeze_support()
    # load the model
    model: CNN = load_model('binary_train/best.model')

    test_files = glob(r'test_data\audio\*.wav')[0:2]
    print(test_files, file=sys.stderr)

    # validation_df = validation_df.head()

    scores_df, preds_df, labels_df = model.predict(num_workers=8,
                                                   samples=test_files,
                                                   split_files_into_clips=True,
                                                   final_clip='full',
                                                   binary_preds=None,
                                                   activation_layer='sigmoid')

    report_file: Path = Path("output.csv")
    with open(report_file, "w") as rf:
        results_df: DataFrame = scores_df
        print(f" filename; begin; end; call_type ; type1  ;type2  ; type3  ; type4", file=rf)
        for row in results_df.itertuples():
            print(
                f" \"{(row[0][0])}\"; {row[0][1]}s ; {row[0][2]}s ; type{(row[0][0])[-6:-5]} ; {row[1]:.2f}  ; {row[2]:.2f}  ; {row[3]:.2f}  ; {row[4]:.2f} ",
                file=rf)
        print(scores_df)
# the prediction DS look like this:
#                                                                               type1  ...         type4
# file                                               start_time end_time                ...
# training_data\chunks\20210418_184500.loc01_00;3... 0.0        3.0       1.000000e+00  ...  1.387167e-14
# training_data\chunks\20210418_184500.loc01_00;3... 0.0        3.0       9.994953e-01  ...  5.046870e-04
