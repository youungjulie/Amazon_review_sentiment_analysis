# import ML models
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from module import model_performance, train_model


# load data
X_train_counts = joblib.load('data/featured/X_train_counts.pkl')
X_train_tfidf = joblib.load('data/featured/X_train_tfidf.pkl')
X_train_avgw2v = joblib.load('data/featured/X_train_avgw2v.pkl')
train_data = pd.read_csv('data/sl_train_data.csv')
y_train = train_data['label']

X_val_counts = joblib.load('data/featured/X_val_counts.pkl')
X_val_tfidf = joblib.load('data/featured/X_val_tfidf.pkl')
X_val_avgw2v = joblib.load('data/featured/X_val_avgw2v.pkl')
val_data = pd.read_csv('data/sl_val_data.csv')
y_val = val_data['label']

X_test_counts = joblib.load('data/featured/X_test_counts.pkl')
X_test_tfidf = joblib.load('data/featured/X_test_tfidf.pkl')
X_test_avgw2v = joblib.load('data/featured/X_test_avgw2v.pkl')
test_data = pd.read_csv('data/sl_test_data.csv')
y_test = test_data['label']

feature_names = ['counts', 'tfidf', 'avgw2v']
X_train = [X_train_counts, X_train_tfidf, X_train_avgw2v]
X_val = [X_val_counts, X_val_tfidf, X_val_avgw2v]
X_test = [X_test_counts, X_test_tfidf, X_test_avgw2v]

print("Count Vectorizer X_train shape: ", X_train_counts.shape)
print("Count Vectorizer y_train shape: ", y_train.shape)
print("TF-IDF Vectorizer X_train shape: ", X_train_tfidf.shape)
print("Avg Word2Vec X_train shape: ", X_train_avgw2v.shape)

# model training
print('=' * 50)
print("Model training...")

# trivial baseline model
# randomly assign labels with class distribution
# calculate the probability of y = 1
print("Trivial baseline model:")
y_prob = y_train.sum() / len(y_train)
y_pred = np.random.choice([0, 1], size=len(y_val), p=[1 - y_prob, y_prob])
trivial_val_acc = (y_pred == y_val).sum() / len(y_val)
print("Validation accuracy: {:.4f}".format(trivial_val_acc))
y_pred = np.random.choice([0, 1], size=len(y_test), p=[1 - y_prob, y_prob])
trivial_test_acc = (y_pred == y_test).sum() / len(y_test)
print("Test accuracy: {:.4f}".format(trivial_test_acc))

# non trivial baseline model
print("="*50)
print("Non-trivial baseline model:")
lrm = LogisticRegression(penalty='l2', C=1, solver='liblinear')
model_performance(lrm, X_train_tfidf, y_train, X_test_tfidf, y_test, validation=False, X_val=X_val_tfidf, y_val=y_val)

# Non-Baseline Models
# Try the following models:
train_model(X_train, y_train, X_val, y_val, "Logistic Regression", feature_names, metric='accuracy')
train_model(X_train, y_train, X_val, y_val, "SVM", feature_names, metric='accuracy')
train_model(X_train, y_train, X_val, y_val, "Random Forest", feature_names, metric='accuracy')
train_model(X_train, y_train, X_val, y_val, "XGBoost", feature_names, metric='accuracy')
print("Model training finished.")


print('=' * 50)
print("Model Selection...")
# trivial baseline model
print("="*50)
print("Trivial baseline model:")
y_prob = y_train.sum() / len(y_train)
y_pred = np.random.choice([0, 1], size=len(y_val), p=[1 - y_prob, y_prob])
trivial_val_acc = (y_pred == y_val).sum() / len(y_val)
print("Accuracy on validation set: {:.4f}".format(trivial_val_acc))
y_pred = np.random.choice([0, 1], size=len(y_test), p=[1 - y_prob, y_prob])
trivial_test_acc = (y_pred == y_test).sum() / len(y_test)
print("Accuracy on test set: {:.4f}".format(trivial_test_acc))



# SVM model
print("="*50)
print("SVM model:")
svm = SVC(C=10, kernel='rbf', gamma='scale', degree=2, tol=0.001, probability=True)
model_performance(svm, X_train_tfidf, y_train, X_test_tfidf, y_test, validation=False, X_val=X_val_tfidf, y_val=y_val)

# Random Forest model
print("="*50)
print("Random Forest model:")
rf = RandomForestClassifier(criterion='entropy', n_estimators=500, max_depth=60, random_state=42)
model_performance(rf, X_train_counts, y_train, X_test_counts, y_test, validation=False, X_val=X_val_counts, y_val=y_val)

# XGBoost model
print("="*50)
print("XGBoost model:")
xgb = XGBClassifier(booster='gbtree', n_estimators=200, learning_rate=0.5,
                    max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                    colsample_bytree=0.8, objective='binary:logistic', nthread=4,
                    scale_pos_weight=1, seed=0)
model_performance(xgb, X_train_counts, y_train, X_test_counts, y_test, validation=False, X_val=X_val_counts, y_val=y_val)


# We need to refine the random forest model with the best parameters
# use the BOW feature to train the model
print("Refine the Random Forest model...")
n_est = [400, 450, 500, ]
max_depth = [60, 65, 70]
val_acc_score_mat = np.zeros((len(n_est), len(max_depth)))
test_acc_score_mat = np.zeros((len(n_est), len(max_depth)))
for i, n in enumerate(n_est):
    for j, d in enumerate(max_depth):
        rf = RandomForestClassifier(criterion='entropy', n_estimators=n, max_depth=d, random_state=42)
        val_acc, test_acc = model_performance(rf, X_train_counts, y_train,
                                              X_test_counts, y_test,
                                              validation=False,
                                              X_val=X_val_counts, y_val=y_val)

        print("n_estimators: {}, max_depth: {}, val_acc: {:.4f}, test_acc: {:.4f}".format(n, d, val_acc, test_acc))
        val_acc_score_mat[i, j] = val_acc
        test_acc_score_mat[i, j] = test_acc

# plot the heatmap of difference between validation and test accuracy
plt.figure(figsize=(10, 8))
sns.heatmap(val_acc_score_mat - test_acc_score_mat, annot=True, fmt='.4f', cmap='Blues')
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.title('Difference between validation and test accuracy')
plt.show()


# store the best model
print('=' * 50)
print("Store the best model...")
best_model_RF = RandomForestClassifier(criterion='entropy', n_estimators=400, max_depth=70, random_state=42)
best_model = best_model_RF
best_model.fit(X_train_tfidf, y_train)
joblib.dump(best_model, 'best_model_SL.pkl')
