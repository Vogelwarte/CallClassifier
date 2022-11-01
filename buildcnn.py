# the cnn module provides classes for training/predicting with various types of CNNs
from matplotlib import pyplot as plt
from matplotlib_inline.config import InlineBackend
from opensoundscape.torch.models.cnn import CNN

#other utilities and packages
import torch
#import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
import subprocess

#set up plotting
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for large visuals
InlineBackend.figure_format = 'retina'
if __name__ =='__main__':
    curlew_table = pd.read_csv(Path("C:\\EcoHack\\classified_tables_audio\\OutPut\\Chunks\\20221101_120529UTC_summary.txt")).set_index('filename')
    #print(curlew_table.to_string())
    #curlew_table.head()
    from opensoundscape.annotations import categorical_to_one_hot
    #one_hot_labels, classes = categorical_to_one_hot(curlew_table[['type1']].values)
    labels = curlew_table.columns.values.tolist()
    #pd.DataFrame(index=curlew_table['filename'], data=one_hot_labels, columns=classes)
    print(str(labels))
    from sklearn.model_selection import train_test_split
    train_df, validation_df = train_test_split(curlew_table, test_size=0.2, random_state=1)
    print(train_df.head())
    print(validation_df.head())
    # Create model object
    classes = train_df.columns
    model = CNN('resnet18', classes=classes, sample_duration=2.0, single_target=True)
    #Logging the Model preformance
    model.logging_level = 3  # request lots of logged content
    model.log_file = './binary_train/training_log.txt'  # specify a file to log output to
    Path(model.log_file).parent.mkdir(parents=True, exist_ok=True)  # make the folder ./binary_train

    model.verbose = 0  # don't print anything to the screen during training


    #Train the Model
    model.train(
        train_df=train_df,
        validation_df=validation_df,
        save_path='./binary_train/', #where to save the trained model
        epochs=5,
        batch_size=8,
        save_interval=5, #save model every 5 epochs (the best model is always saved in addition)
        num_workers=12, #specify 4 if you have 4 CPU processes, eg; 0 means only the root process
    )
    #Plot the Loss History
    plt.scatter(model.loss_hist.keys(), model.loss_hist.values())
    plt.xlabel('epoch')
    plt.ylabel('loss')
