"""
This file is for loading data and outputs test result.
should be located in the base directory of your extracted zip folder.

read the data files in “data” folder and output the results
of your final system in your report.

You should not let “main” file to start training. “main” file should load trained
parameters and reproduce the test results
"""
# import ML models
from pprint import pprint

import joblib
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from module import model_performance

# load data
X_train_counts = joblib.load('data/featured/X_train_counts.pkl').toarray()
train_data = pd.read_csv('data/sl_train_data.csv')
y_train = train_data['label'].tolist()

X_val_counts = joblib.load('data/featured/X_val_counts.pkl').toarray()
val_data = pd.read_csv('data/sl_val_data.csv')
y_val = val_data['label'].tolist()

X_test_counts = joblib.load('data/featured/X_test_counts.pkl').toarray()
test_data = pd.read_csv('data/sl_test_data.csv')
y_test = test_data['label'].tolist()

# count vectorizer
target_labeled = pd.read_csv('data/target_labeled_data.csv')
y_tl = target_labeled['label'].tolist()
X_tl_counts = joblib.load('data/featured_tl/X_tl_counts.pkl').toarray()
X_tl_cnt_train, X_tl_cnt_test_temp, y_tl_cnt_train, y_tl_cnt_test_temp = train_test_split(X_tl_counts, y_tl,
                                                                                          test_size=0.25,
                                                                                          random_state=42)
X_tl_cnt_val, X_tl_cnt_test, y_tl_cnt_val, y_tl_cnt_test = train_test_split(X_tl_cnt_test_temp,
                                                                            y_tl_cnt_test_temp,
                                                                            test_size=0.5,
                                                                            random_state=42)

# load the best model
print("====== Best Supervised Learning Model ======")
print("Random Forest:")
best_sl_model = joblib.load('best_model.pkl')
print("Best model parameters: ")
pprint(best_sl_model.get_params())
model_performance(best_sl_model,
                  X_train=X_train_counts, y_train=y_train,
                  X_test=X_test_counts, y_test=y_test,
                  validation=False,
                  X_val=X_val_counts, y_val=y_val)

# load the best model
print("====== Best Transfer Learning Model ======")
best_tl_model = joblib.load('best_model_TL.pkl')
print("Best model parameters: ")
pprint(best_tl_model.get_params())
val_score = best_tl_model.score(X_tl_cnt_val, y_tl_cnt_val)
test_score = best_tl_model.score(X_tl_cnt_test, y_tl_cnt_test)
print("Accuracy on validation set: {:.4f}".format(val_score))
print("Accuracy on test set: {:.4f}".format(test_score))


