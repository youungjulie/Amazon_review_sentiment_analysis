from adapt.feature_based import SA
from adapt.instance_based import TrAdaBoost
import numpy as np


def standardize_my(X, mu=None, std=None):
    # standardize the data X with provided mu and std.
    # if mu or std is None, calculate mu and std from X.
    # X: [num_of_sample, feat_dim]
    if mu is None or std is None:
        mu = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
    X = (X - mu) / (std + 1e-8)
    return X, mu, std


def train_transfer_SA(model, feature_names, X_target_unlabeled, X_source_labeled, y_source_labeled,
                      X_target_labeled_train, X_target_labeled_val,
                      y_target_labeled_train, y_target_labeled_val, n_components=100):
    scores = []
    for i in range(len(feature_names)):
        print('=' * 50)
        print(f"Feature: {feature_names[i]}")
        print('-' * 50)
        SA_model = SA(model, X_target_unlabeled[i], random_state=42, n_components=n_components)
        SA_model.fit(X_source_labeled[i], y_source_labeled[i],
                     X_target_labeled_train[i], y_target_labeled_train[i])

        train_score = round(SA_model.score(X_target_labeled_train[i], y_target_labeled_train[i]), 6)
        val_score = round(SA_model.score(X_target_labeled_val[i], y_target_labeled_val[i]), 6)
        print("Score on training set: ", train_score)
        print("Score on val set: ", val_score)

        # train scores
        scores.append([train_score, val_score])
    return scores


def train_transfer_TrAda(model, feature_names,
                         X_target_labeled_train, y_target_labeled_train,
                         X_source_labeled, y_source_labeled,
                         X_target_labeled_val, y_target_labeled_val,
                         X_target_labeled_test, y_target_labeled_test):
    scores = []
    for i in range(len(feature_names)):
        print('=' * 50)
        print(f"Feature: {feature_names[i]}")
        print('-' * 50)
        TrAdaBoost_model = TrAdaBoost(estimator=model,
                                      Xt=X_target_labeled_train[i],
                                      yt=y_target_labeled_train[i],
                                      n_estimators=10, lr=0.5,
                                      random_state=42)
        TrAdaBoost_model.fit(X_source_labeled[i], y_source_labeled[i])

        val_score = round(TrAdaBoost_model.score(X_target_labeled_val[i], y_target_labeled_val[i]), 6)
        test_score = round(TrAdaBoost_model.score(X_target_labeled_test[i], y_target_labeled_test[i]), 6)
        print("Score on val set: ", val_score * 100, "%")
        print("Score on test set: ", test_score * 100, "%")

        # train scores
        scores.append([val_score, test_score])
    return scores
