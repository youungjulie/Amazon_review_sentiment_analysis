# import ML models
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from adapt.instance_based import TrAdaBoost
from adapt.feature_based import SA
# referenced the library and the map from the web page:
# https://adapt-python.github.io/adapt/generated/adapt.feature_based.SA.html#adapt.feature_based.SA.transform
# https://adapt-python.github.io/adapt/map.html
from module_tl import train_transfer_SA, train_transfer_TrAda
from sklearn.preprocessing import StandardScaler

# load dataset
source_labeled = pd.read_csv('data/source_labeled_data.csv')
source_unlabeled = pd.read_csv('data/source_unlabeled_data.csv')
target_labeled = pd.read_csv('data/target_labeled_data.csv')
target_unlabeled = pd.read_csv('data/target_unlabeled_data.csv')

# load labels y
y_sl = source_labeled['label'].tolist()
y_tl = target_labeled['label'].tolist()

# load features X
# count vectorizer
X_sl_counts = joblib.load('data/featured_tl/X_sl_counts.pkl').toarray()
X_sul_counts = joblib.load('data/featured_tl/X_sul_counts.pkl').toarray()
X_tl_counts = joblib.load('data/featured_tl/X_tl_counts.pkl').toarray()
X_tul_counts = joblib.load('data/featured_tl/X_tul_counts.pkl').toarray()

# tfidf vectorizer
X_sl_tfidf = joblib.load('data/featured_tl/X_sl_tfidf.pkl').toarray()
X_sul_tfidf = joblib.load('data/featured_tl/X_sul_tfidf.pkl').toarray()
X_tl_tfidf = joblib.load('data/featured_tl/X_tl_tfidf.pkl').toarray()
X_tul_tfidf = joblib.load('data/featured_tl/X_tul_tfidf.pkl').toarray()

# train test split for target labeled data
# train set: 450 samples
# validation set: 75 samples
# test set: 75 samples

# count vectorizer
X_tl_cnt_train, X_tl_cnt_test_temp, y_tl_cnt_train, y_tl_cnt_test_temp = train_test_split(X_tl_counts, y_tl,
                                                                                          test_size=0.25,
                                                                                          random_state=42)
X_tl_cnt_val, X_tl_cnt_test, y_tl_cnt_val, y_tl_cnt_test = train_test_split(X_tl_cnt_test_temp,
                                                                            y_tl_cnt_test_temp,
                                                                            test_size=0.5,
                                                                            random_state=42)
# tfidf vectorizer
X_tl_tf_train, X_tl_tf_test_temp, y_tl_tf_train, y_tl_tf_test_temp = train_test_split(X_tl_tfidf, y_tl,
                                                                                      test_size=0.25,
                                                                                      random_state=42)
X_tl_tf_val, X_tl_tf_test, y_tl_tf_val, y_tl_tf_test = train_test_split(X_tl_tf_test_temp, y_tl_tf_test_temp,
                                                                        test_size=0.5,
                                                                        random_state=42)

# trivial transfer learning
print("=" * 20 + "Domain Adaptation Transfer Learning" + "=" * 20)
print("=" * 20 + "Trivial Transfer Learning" + "=" * 20)
# count vectorizer
# train RF model
simple_clf = SVC(C=0.5, kernel='linear', gamma='scale', probability=True)
simple_clf.fit(X_tl_cnt_train, y_tl_cnt_train)
clf_cnt = RandomForestClassifier(criterion="entropy", n_estimators=400, max_depth=70, random_state=42)
clf_cnt.fit(X_tl_cnt_train, y_tl_cnt_train)
# validation accuracy
rf_cnt_val_acc = clf_cnt.score(X_tl_cnt_val, y_tl_cnt_val)
print('Random Forest count vectorizer validation accuracy: ', rf_cnt_val_acc)
# test accuracy
rf_cnt_test_acc = clf_cnt.score(X_tl_cnt_test, y_tl_cnt_test)
print('Random Forest count vectorizer test accuracy: ', rf_cnt_test_acc)

# domain apaptation transfer learning
feature_names = ['counts', 'tfidf']
# source labeled data
X_sl = [X_sl_counts, X_sl_tfidf]
y_sl_set = [y_sl, y_sl]

# target labeled data
X_tl_train = [X_tl_cnt_train, X_tl_tf_train]
X_tl_val = [X_tl_cnt_val, X_tl_tf_val]
X_tl_test = [X_tl_cnt_test, X_tl_tf_test]
y_tl_train = [y_tl_cnt_train, y_tl_tf_train]
y_tl_val = [y_tl_cnt_val, y_tl_tf_val]
y_tl_test = [y_tl_cnt_test, y_tl_tf_test]
# target unlabeled data
X_tul = [X_tul_counts, X_tul_tfidf]

# train model

simple_clf = SVC(C=0.5, kernel='linear', gamma='scale', probability=True)
rf_clf = RandomForestClassifier(criterion='entropy', max_depth=40, n_estimators=700)
#
# print('=' * 50)
# print("Model training...")
#
# count_acc = []
# tfidf_acc = []
# for n in range(200, 400, 50):
#     # When using the source and domain dataset without standarization, the accuracy of the model is around 0.65 +/-
#     scores = train_transfer(rf_clf, feature_names, X_tl, X_sl, y_sl_set,
#                             X_tl_train, X_tl_val,
#                             y_tl_train, y_tl_val, n_components=n)
#     count_score = scores[0]
#     tf_score = scores[1]
#     count_acc.append(count_score)
#     tfidf_acc.append(tf_score)
#
# print("Model training finished!")
#
# # plot the accuracy of the model
#
#
# model_name = "Random Forest"
# # plot the train and validation accuracy
# plt.plot(range(200, 400, 50), [acc[0] for acc in count_acc], label='train')
# plt.plot(range(200, 400, 50), [acc[1] for acc in count_acc], label='validation')
# plt.title(model_name + ' Count Vectorizer Accuracy')
# plt.xlabel('Number of Components')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# plt.plot(range(10, 110, 20), [acc[0] for acc in tfidf_acc], label='train')
# plt.plot(range(10, 110, 20), [acc[1] for acc in tfidf_acc], label='validation')
# plt.xlabel('Number of components')
# plt.ylabel('Accuracy')
# plt.title('Accuracy of TFIDF Vectorizer, {}'.format(model_name))
# plt.legend()
# plt.show()

# To optimize the performance, let's try the standardized data
# standardize the data for source and target domain
print("Standardize the data for source and target domain...")
print("for count vectorizer...")
# count vectorizer
# source domain
sl_counts_std_scaler = StandardScaler()
sl_counts_std_scaler.fit(X_sl_counts)
X_sl_counts_std = sl_counts_std_scaler.transform(X_sl_counts)
# target domain
tl_counts_std_scaler = StandardScaler(with_mean=True, with_std=True)
tl_counts_std_scaler.fit(X_tul_counts)
X_tl_cnt_train_std = tl_counts_std_scaler.transform(X_tl_cnt_train)
X_tl_cnt_val_std = tl_counts_std_scaler.transform(X_tl_cnt_val)
X_tl_cnt_test_std = tl_counts_std_scaler.transform(X_tl_cnt_test)

print("for tfidf vectorizer...")
# tfidf vectorizer
# source domain
sl_tfidf_std_scaler = StandardScaler()
sl_tfidf_std_scaler.fit(X_sl_tfidf)
X_sl_tfidf_std = sl_tfidf_std_scaler.transform(X_sl_tfidf)
# target domain
tl_tfidf_std_scaler = StandardScaler(with_mean=True, with_std=True)
tl_tfidf_std_scaler.fit(X_tul_tfidf)
X_tl_tf_train_std = tl_tfidf_std_scaler.transform(X_tl_tf_train)
X_tl_tf_val_std = tl_tfidf_std_scaler.transform(X_tl_tf_val)
X_tl_tf_test_std = tl_tfidf_std_scaler.transform(X_tl_tf_test)

# combine the count and tfidf vectorizer standardize data
X_sl_std = [X_sl_counts_std, X_sl_tfidf_std]
X_tl_train_std = [X_tl_cnt_train_std, X_tl_tf_train_std]
X_tl_val_std = [X_tl_cnt_val_std, X_tl_tf_val_std]
X_tl_test_std = [X_tl_cnt_test_std, X_tl_tf_test_std]

feature_names_cnt = ['counts']
# train model
print('=' * 50)
print("Model training...")
print("Subspace alignment...")
train_transfer_SA(rf_clf, feature_names_cnt, X_tl_train_std, X_sl_std, y_sl_set,
                  X_tl_train_std, X_tl_val_std,
                  y_tl_train, y_tl_val, n_components=150)

print("TrAdaBoost...")
print("source domain data shape: X={}, y={}".format(X_sl[0].shape, len(y_sl_set[0])))
print("target domain data shape: X={}, y={}".format(X_tl_train[0].shape, len(y_tl_train[0])))
train_transfer_TrAda(rf_clf, feature_names_cnt,
                     X_tl_train, y_tl_train,
                     X_sl, y_sl_set,
                     X_tl_val, y_tl_val,
                     X_tl_test, y_tl_test)

# Best Model Export TrAdaBoost
n_estimators = 10
learning_rate = 0.5
TrAdaBoost_clf = TrAdaBoost(estimator=rf_clf,
                            Xt=X_tl_cnt_train,
                            yt=y_tl_cnt_train,
                            n_estimators=n_estimators, lr=learning_rate,
                            random_state=42)
TrAdaBoost_clf.fit(X_sl_counts, y_sl)
joblib.dump(TrAdaBoost_clf, 'best_model_TL.pkl')
