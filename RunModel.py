from pathlib import Path
import pandas as pd
from opensoundscape.torch.models.cnn import load_model


#Setup Data for Model
from pandas import DataFrame

curlew_table = pd.read_csv(Path(r'training_data\chunks\_summary.csv')).set_index('filename')
#print(curlew_table.to_string())
#curlew_table.head()
from opensoundscape.annotations import categorical_to_one_hot
#one_hot_labels, classes = categorical_to_one_hot(curlew_table[['type1']].values)
labels = curlew_table.columns.values.tolist()
#pd.DataFrame(index=curlew_table['filename'], data=one_hot_labels, columns=classes)
print(str(labels))
from sklearn.model_selection import train_test_split
train_df, validation_df = train_test_split(curlew_table, test_size=0.2, random_state=1)
#print(train_df.head())
#print(validation_df.head())


#Run Model
model = load_model('binary_train/best.model')
#%%
#create a copy of the training dataset, sampling 0 of the training samples from it
#prediction_dataset = model.train_dataset.sample(n=0)
#turn off augmentation on this dataset
#prediction_dataset.augmentation_off()
#use the validation samples as test samples for the sake of illustration
#prediction_dataset.df = validation_df
# ### Predict on the validation dataset
#
# We simply call model's `.predict()` method on a Preprocessor instance.
#
# This will return three dataframes:
#
# - scores : numeric predictions from the model for each sample and class (by default these are raw outputs from the model)
# - predictions: 0/1 predictions from the model for each sample and class (only generated if `binary_predictions` argument is supplied)
# - labels: Original labels from the dataset, if available
#
# In[19]:
#valid_scores_df, valid_preds_df, valid_labels_df = model.predict(prediction_dataset, binary_preds='multi_target', activation_layer='sigmoid')

validation_df = validation_df.head()
valid_scores_df, valid_preds_df, valid_labels_df = model.predict(validation_df, binary_preds='single_target', activation_layer='softmax')

report_file: Path = "output.csv"
with open(report_file, "w") as rf:
    results_df:DataFrame = valid_scores_df
    print(f" filename; begin; end; call_type ; type1  ;type2  ; type3  ; type4", file=rf)
    for row in results_df.itertuples():
        print(f" \"{(row[0][0])}\"; {row[0][1]}s ; {row[0][2]}s ; type{(row[0][0])[-6:-5]} ; {row[1]:.2f}  ; {row[2]:.2f}  ; {row[3]:.2f}  ; {row[4]:.2f} ", file=rf)

#print(valid_scores_df.to_string())
#print (results_df.to_string(