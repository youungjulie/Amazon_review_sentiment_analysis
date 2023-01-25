import re
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Training model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from adapt.instance_based import TrAdaBoost
# https://adapt-python.github.io/adapt/generated/adapt.instance_based.TrAdaBoost.html

# evaluate metrics
from sklearn.metrics import roc_curve, auc


# data format
# The preprocessed data is one line per document, with each line in the format:
# feature:<count> .... feature:<count> #label#:<label>

def read_data(file):
    data = []
    count = 1
    for line in file:
        line = line.strip()  # remove the trailing newline
        if line:
            features, label = line.split('#label#')
            label = label.split(':')[1]
            features = features.split(' ')
            features = [f.split(':') for f in features]
            features = {f[0]: int(f[1]) for f in features if len(f) == 2}
            # convert positive to 1 and negative to 0
            label = 1 if label == 'positive' else 0
            data.append((features, label))
    return data


def join_sentence(sentences):
    for i, s in enumerate(sentences):
        sentences[i] = ' '.join(s).replace('_', ' ')
    return sentences


def create_df(pos_data, neg_data):
    # tokenize the words
    pos_sentences = []
    for words, label in pos_data:
        sentence = []
        for word, cnt in words.items():
            sentence.extend([word] * cnt)
        pos_sentences.append(sentence)
    neg_sentences = []
    for words, label in neg_data:
        sentence = []
        for word, cnt in words.items():
            sentence.extend([word] * cnt)
        neg_sentences.append(sentence)

    pos_sentences = join_sentence(pos_sentences)
    neg_sentences = join_sentence(neg_sentences)
    df = pd.DataFrame(columns=['sentence', 'label'])
    df['sentence'] = pos_sentences + neg_sentences
    df['label'] = [1] * len(pos_sentences) + [0] * len(neg_sentences)

    return df


def create_df_unlabeled(unlabeled_data):
    # tokenize the words
    unlabeled_sentences = []
    for words, label in unlabeled_data:
        sentence = []
        for word, cnt in words.items():
            sentence.extend([word] * cnt)
        unlabeled_sentences.append(sentence)

    unlabeled_sentences = join_sentence(unlabeled_sentences)

    df = pd.DataFrame(columns=['sentence', 'label'])
    df['sentence'] = unlabeled_sentences
    df['label'] = [label for words, label in unlabeled_data]
    return df


# feature preparation
def clean_words(X_train):
    print("Remove punctuation...")
    X_train_no_punc = X_train.apply(remove_punc)  # remove punctuation
    print("Tokenization...")
    X_train_tokens = [word_tokenize(x) for x in X_train_no_punc]  # tokenize
    # print("Remove stop words...")
    # X_train_clean = []
    # for tokens in tqdm(X_train_tokens):
    #     X_train_clean.append([w for w in tokens if w not in stopwords.words('english')])
    return X_train_tokens


def remove_punc(text):
    '''
    text: a string
    '''
    text = re.sub(r'[^\w\s]', '', text)
    return text


# feature extraction Thanks for the code from
# https://github.com/Sachin-D-N/Amazon_Food_Reviews/blob/main/08.Random_Forest_Amazon_Food_Reviews/Random_Forest_Amazon_Food_Reviews_Assignment_.ipynb
def avg_w2vector(reviews, w2v_words, model, num_features):
    # average Word2Vec
    # compute average word2vec for each review.
    review_vects = []  # the avg-w2v for each sentence/review is stored in this list
    for rev in tqdm(reviews):  # for each review/sentence
        rev_vec = np.zeros(num_features)  # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v
        cnt_words = 0  # num of words with a valid vector in the sentence/review
        for word in rev:  # for each word in a review/sentence
            if word in w2v_words:
                vec = model.wv[word]
                rev_vec += vec
                cnt_words += 1
        if cnt_words != 0:
            rev_vec /= cnt_words
        review_vects.append(rev_vec)
    review_vects = np.array(review_vects)
    print("Review shape (# of reviews, # of features)", review_vects.shape)
    return review_vects


def train_model(X_train, y_train, X_val, y_val, model_name, feature_names, metric='accuracy'):
    print("#### ", model_name, " ####")
    if model_name == 'Logistic Regression':
        for feature in feature_names:
            print("\n#", feature, "Vectorizer...")
            p, c = grid_search_LR(X_train[feature_names.index(feature)], y_train,
                                  metric=metric,
                                  model=LogisticRegression(solver='liblinear'))
            print("** Model performance on validation set...")
            model_performance(LogisticRegression(penalty=p, C=c, solver='liblinear'),
                              X_train=X_train[feature_names.index(feature)], y_train=y_train,
                              X_test=X_val[feature_names.index(feature)], y_test=y_val,
                              validation=True)
            print("Done!")

    elif model_name == 'SVM':
        for feature in feature_names:
            print("\n#", feature, "Vectorizer...")
            c, k = grid_search_SVM(X_train[feature_names.index(feature)], y_train,
                                   metric=metric,
                                   model=SVC(degree=2, gamma='scale', probability=True, tol=0.001))
            print("** Model performance on validation set...")
            model_performance(SVC(C=c, kernel=k, degree=2, gamma='scale', probability=True, tol=0.001),
                              X_train=X_train[feature_names.index(feature)], y_train=y_train,
                              X_test=X_val[feature_names.index(feature)], y_test=y_val,
                              validation=True)

    elif model_name == 'Random Forest':
        for feature in feature_names:
            print("\n#", feature, "Vectorizer...")
            n, d = grid_search_RF(X_train[feature_names.index(feature)], y_train,
                                  metric=metric,
                                  model=RandomForestClassifier(criterion='entropy'))
            print("** Model performance on validation set...")
            model_performance(RandomForestClassifier(n_estimators=n, max_depth=d, criterion='entropy'),
                              X_train=X_train[feature_names.index(feature)], y_train=y_train,
                              X_test=X_val[feature_names.index(feature)], y_test=y_val,
                              validation=True)

            print("Done!")

    elif model_name == 'XGBoost':
        for feature in feature_names:
            print("\n#", feature, "Vectorizer...")
            eta, d = grid_search_XG(X_train[feature_names.index(feature)], y_train,
                                  metric=metric)
            print("** Model performance on validation set...")
            model_performance(XGBClassifier(learning_rate=eta, max_depth=d, booster='gbtree', n_estimators=200),
                              X_train=X_train[feature_names.index(feature)], y_train=y_train,
                              X_test=X_val[feature_names.index(feature)], y_test=y_val,
                              validation=True)

            print("Done!")

    print("Finished training for", model_name, "!")


def grid_search_LR(X_train, y_train, metric='accuracy', model=LogisticRegression(solver='liblinear')):
    penalty = ['l1', 'l2']
    C = [0.1, 1, 10]

    param_grid = {'penalty': penalty, 'C': C}
    clf = GridSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1, pre_dispatch=2, return_train_score=True)
    clf.fit(X_train, y_train)

    print('****', metric, 'score for CV Logistic Regression****')
    print('optimal penalty: ', clf.best_params_['penalty'])
    print('optimal C: ', clf.best_params_['C'])
    print('Best', metric, 'score: ', clf.best_score_)

    sns.set()
    df_grid = pd.DataFrame(clf.cv_results_)
    max_scores = df_grid.groupby(['param_penalty', 'param_C']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g', vmin=0.65, vmax=0.88)
    plt.xlabel('C')
    plt.ylabel('penalty')
    title = metric + ' score for CV Logistic Regression'
    plt.title(title)
    plt.show()

    # save the image
    plt.savefig('model_result/accuracy_score_for_CV_Logistic_Regression.png')

    return clf.best_params_['penalty'], clf.best_params_['C']


def grid_search_SVM(X_train, y_train, metric='roc_auc',
                    model=SVC(degree=2, gamma='scale', probability=True, tol=0.01)):
    C = [0.5, 1, 10]
    kernel = ['linear', 'rbf']

    param_grid = {'C': C, 'kernel': kernel}
    clf = GridSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1, pre_dispatch=2, return_train_score=True)
    clf.fit(X_train, y_train)

    print('****', metric, 'score for CV SVM****')
    print('optimal C: ', clf.best_params_['C'])
    print('optimal kernel: ', clf.best_params_['kernel'])
    print('Best', metric, 'score: ', clf.best_score_)

    sns.set()
    df_grid = pd.DataFrame(clf.cv_results_)
    max_scores = df_grid.groupby(['param_C', 'param_kernel']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g', vmin=0.65, vmax=0.88)
    plt.xlabel('kernel')
    plt.ylabel('C')
    title = metric + ' score for CV SVM '
    plt.title(title)
    plt.show()

    # save the image
    plt.savefig('model_result/accuracy_score_for_CV_SVM.png')

    return clf.best_params_['C'], clf.best_params_['kernel']


def grid_search_RF(X_train, y_train, metric='roc_auc', model=RandomForestClassifier(criterion='entropy')):
    n_est = [200, 300, 400, 500]
    max_depth = [20, 40, 60, 80]

    param_grid = {'n_estimators': n_est, 'max_depth': max_depth, }
    clf = GridSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1, pre_dispatch=2, return_train_score=True)
    clf.fit(X_train, y_train)

    print('**** ', metric, ' score for CV Random Forest ****')
    print('optimal max_depth: ', clf.best_params_['max_depth'])
    print('optimal n_estimators: ', clf.best_params_['n_estimators'])
    print('best ', metric, 'score: ', clf.best_score_)

    sns.set()
    df_grid = pd.DataFrame(clf.cv_results_)
    max_scores = df_grid.groupby(['param_max_depth', 'param_n_estimators']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g', vmin=0.75, vmax=0.88)
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    title = metric + ' score for CV Random Forest'
    plt.title(title)
    plt.savefig('model_result/accuracy_score_for_CV_Random_Forest.png')
    plt.show()

    # save the image

    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']


def grid_search_XG(X_train, y_train, metric='accuracy', model=XGBClassifier(booster='gbtree', n_estimators=70)):
    eta = [0.4, 0.5, 0.6, 0.7]
    max_depth = [20, 25, 30]

    param_grid = {'max_depth': max_depth, 'learning_rate': eta}
    clf = GridSearchCV(model, param_grid, cv=3, scoring=metric, n_jobs=-1, pre_dispatch=2, return_train_score=True)
    clf.fit(X_train, y_train)

    print('****', metric, ' score for CV XGBRFClassifier****')
    print('optimal max_depth: ', clf.best_params_['max_depth'])
    print('optimal learning rate: ', clf.best_params_['learning_rate'])
    print('best ', metric, 'score: ', clf.best_score_)

    sns.set()
    df_grid = pd.DataFrame(clf.cv_results_)
    max_scores = df_grid.groupby(['param_max_depth', 'param_learning_rate']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g', vmin=0.75, vmax=0.86)
    plt.xlabel('learning_rate')
    plt.ylabel('max_depth')
    title = metric + ' score for CV XGBClassifier'
    plt.title(title)
    plt.savefig('model_result/accuracy_score_for_CV_XGBoost.png')
    plt.show()

    # save the image

    return clf.best_params_['learning_rate'], clf.best_params_['max_depth']


def model_performance(best_model, X_train, y_train, X_test, y_test, validation=False, X_val=None, y_val=None):
    best_model.fit(X_train, y_train)

    scores = []
    # predict accuracy
    if validation:
        print("Accuracy on training set: {:.4f}".format(best_model.score(X_train, y_train)))
        print("Accuracy on validation set: {:.4f}".format(best_model.score(X_test, y_test)))
    else:
        if X_val is not None:
            val_score = best_model.score(X_val, y_val)
            test_score = best_model.score(X_test, y_test)
            print("Accuracy on validation set: {:.4f}".format(val_score))
            print("Accuracy on test set: {:.4f}".format(test_score))

            scores.append(val_score)
            scores.append(test_score)

        else:
            print("Accuracy on training set: {:.4f}".format(best_model.score(X_train, y_train)))
            print("Accuracy on test set: {:.4f}".format(best_model.score(X_test, y_test)))

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs
    train_fpr, train_tpr, thresholds = roc_curve(y_train, best_model.predict_proba(X_train)[:, 1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])

    print('The AUC_score of data is :', auc(test_fpr, test_tpr))

    # sns.set()
    # plt.plot(train_fpr, train_tpr, label="train AUC =" + str(auc(train_fpr, train_tpr)))
    # plt.plot(test_fpr, test_tpr, label="test AUC =" + str(auc(test_fpr, test_tpr)))
    # plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    # plt.legend()
    # plt.xlabel("False_positive_rate")
    # plt.ylabel("True positive_rate")
    # plt.title("ROC_Curve")
    # plt.grid()
    # plt.savefig('model_result/ROC_Curve.png')
    # plt.show()

    return scores

    # save the image
