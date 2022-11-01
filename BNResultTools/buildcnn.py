# the cnn module provides classes for training/predicting with various types of CNNs
from opensoundscape.torch.models.cnn import CNN

#other utilities and packages
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import random
import subprocess

#set up plotting
#from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize']=[15,5] #for large visuals
#%config InlineBackend.figure_format = 'retina'

curlew_table = pd.read_csv(Path("C:\EcoHack\classified_tables_audio\OutPut\Chunks\20221101_120529UTC_summary.txt"))
curlew_table.head()