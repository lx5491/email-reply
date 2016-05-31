from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import json
import numpy as np

def grab_data(filename):
    emails = []
    count = 0
    with open(filename, "r") as in_file:
        for line in in_file:
            # if count == 200:
            #     break
            msg = dict({})
            msg = json.loads(line)
            emails.append(msg)
            count += 1

    return emails

def preprocess(emails):
    X = []
    Y = []

    word_dict = dict({})
    emails_bag_of_words = []
    count = 0
    index = 0
    for email in emails:
        body_text = email["body"]
        body_words_lst = body_text.split()
        emails_bag_of_words.append(body_words_lst)
        for word in body_words_lst:
            if word not in word_dict:
                word_dict[word] = index
                X.append(0)
                index += 1

        y = 1 if email["got_reply"] == "yes" else -1
        Y.append(y)

        count += 1

    Y = np.array(Y)
    X = np.zeros((len(emails), index), dtype=np.float)

    for i in xrange(len(emails_bag_of_words)):
        bag = emails_bag_of_words[i]
        for word in bag:
            j = word_dict[word]
            X[i][j] += 1

    print "X.shape:", X.shape
    print "Y length:", len(Y)

    num_ones = np.where(Y == 1)[0].shape[0]
    num_neg_ones = np.where(Y == -1)[0].shape[0]
    print "Y has %d 1s and %d -1s" % (num_ones, num_neg_ones)

    print "Scaling"
    X = scale(X)

    return X, Y

def scale(X):
    max_per_col = X.max(axis=0)
    min_per_col = X.min(axis=0)
    print "max_per_col, min_per_col shape:", max_per_col.shape, min_per_col.shape
    max_mat = np.tile(max_per_col, (X.shape[0], 1))
    min_mat = np.tile(min_per_col, (X.shape[0], 1))
    print "max_mat, min_mat shape:", max_mat.shape, min_mat.shape
    
    min_max_sum_mat = max_mat + min_mat
    min_max_diff_mat = max_mat - min_mat
    min_max_diff_mat = np.clip(min_max_diff_mat, 1, float("inf")) # make all the 0 elements to 1
    print min_max_diff_mat.shape
    scaled_X = (2 * X - min_max_sum_mat) / min_max_diff_mat

    # print scaled_X

    return scaled_X

def metric_predict(y_true, y_predict):
    metric = dict({})
    correct = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in xrange(len(y_true)):
        if y_true[i] == y_predict[i] and y_predict[i] == 1:
            true_positive += 1
        elif y_true[i] == y_predict[i] and y_predict[i] == -1:
            true_negative += 1
        elif y_true[i] != y_predict[i] and y_predict[i] == 1:
            false_positive += 1
        elif y_true[i] != y_predict[i] and y_predict[i] == -1:
            false_negative += 1
    print true_positive, true_negative, false_positive, false_negative
    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive) / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)
    metric = {"precision":precision, "recall":recall, "f1_score":f1_score, \
        "tp":true_positive, "fp":false_positive, "tn":true_negative, "fn":false_negative}
    return metric

if __name__ == "__main__":
    t0 = time()
    print "Program starts"
    print "Reading email data..."
    emails = grab_data("email_data_text.txt")

    print "Preprocessing data..."
    t1 = time()
    X, Y = preprocess(emails)
    t2 = time()
    print "Preprocessing finished using %0.3f secs" % (t2 - t1)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=24)
    # print "Train size:", len(y_train)
    # print "Test size:", len(y_test)


    # clf = SVC(C=1, gamma=0.05)
    # clf = clf.fit(X_train, y_train)
    # y_predict_train = clf.predict(X_train)
    # y_predict = clf.predict(X_test)
    # t3 = time()
    # print "Training finished using %0.3f secs" % (t3 - t2)

    # metric_train = metric_predict(y_train, y_predict_train)
    # metric_test = metric_predict(y_test, y_predict)

    # train_precision = metric_train["precision"]
    # train_recall = metric_train["recall"]
    # train_f1_score = metric_train["f1_score"]
    # print "Train precision:", metric_train["tp"], "/", (metric_train["tp"] + metric_train["fp"]), ",", train_precision
    # print "Train recall: ", metric_train["tp"], "/", (metric_train["tp"] + metric_train["fn"]), ",", train_recall
    # print "Train F1 score:", train_f1_score

    # test_precision = metric_test["precision"]
    # test_recall = metric_test["recall"]
    # test_f1_score = metric_test["f1_score"]
    # print "Test precision:", metric_test["tp"], "/", (metric_test["tp"] + metric_test["fp"]), ",", test_precision
    # print "Test recall: ", metric_test["tp"], "/", (metric_test["tp"] + metric_test["fn"]), ",", test_recall
    # print "Test F1 score:", test_f1_score

    # print "Program ends (%0.3f secs)" % (time() - t0)






