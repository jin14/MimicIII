import re
import pandas as pd
import numpy as np

#TFIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import lightgbm as lgb
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle


def trainLightGBM(X_train, X_eval, Y_train, Y_eval, parameters):
    lgb_train = lgb.Dataset(X_train,Y_train)
    lgb_eval = lgb.Dataset(X_eval, Y_eval, reference=lgb_train)

    boosting = ['dart', 'gbdt']
    learning_rate = [0.01, 0.001, 0.0001]
    num_leaves = [50, 80, 100, 120]
    best = float('inf')
    bestmodel = None
    for boost in boosting:
        for lr in learning_rate:
            for nl in num_leaves:
                print("Parameters: ",boost, " " ,lr, " ",nl)
                parameters['num_leaves'] = nl
                parameters['learning_rate'] = lr
                parameters['boosting'] = boost
                lgbm_classfier = lgb.train(parameters, lgb_train, num_boost_round=30, valid_sets=lgb_eval,
                verbose_eval=True)
            score = list(lgbm_classfier.best_score['valid_0'].values())[0]
            if score < best:
                best = score
                bestmodel = lgbm_classfier
    # lgbm_classfier = lgb.train(parameters, lgb_train, num_boost_round=30, valid_sets=lgb_eval,
    #              verbose_eval=True)
    return bestmodel

def evaluateGBModel(model, X_test, Y_test, classes, name):

    fpr, tpr, roc_auc = {}, {} ,{}
    y_pred = model.predict(X_test)
    y_test = Y_test

    for n in range(len(classes)):
        fpr[n], tpr[n], _ = roc_curve(y_test[:, n], y_pred[:, n])
        roc_auc[n] = auc(fpr[n], tpr[n])

    plt.figure()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Baseline")
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'black', 'navy'])
    for i, color in zip(classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, linestyle='--',
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig("../results/ROCcurve_{}.png".format(name), bbox_inches='tight')

    precision, recall, average_precision, auc_scores = {}, {}, {}, {}
    for n in range(len(classes)):
        precision[n], recall[n], _ = precision_recall_curve(y_test[:, n], y_pred[:, n])
        average_precision[n] = average_precision_score(y_test[:, n], y_pred[:, n], average='weighted')
        print('{} Average Precision is {}'.format(n, average_precision[n]))


    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'black', 'navy'])
    lines = []
    labels = []
    plt.figure()
    plt.style.use('ggplot')

    for i, color in zip(classes, colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append("{} AP: {:0.2f}".format(i, average_precision[i]))

    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(lines, labels, title="PR-AUC", loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.savefig("../results/PR_{}.png".format(name), bbox_inches='tight')

    print('Precision Recall Curve generated!')


    return roc_auc

def evaluateGBModelsingle(model, X_test, Y_test, classes, name):

    fpr, tpr, roc_auc = {}, {} ,{}
    y_pred = model.predict(X_test)
    y_test = Y_test
    df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    df.to_csv("../results/results_baseline_{}.csv".format(name), index=False)

    for n in range(len(classes)):
        fpr[n], tpr[n], _ = roc_curve(y_test, y_pred)
        roc_auc[n] = auc(fpr[n], tpr[n])

    plt.figure()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Baseline")
    colors = cycle(['aqua'])
    print(colors,classes)
    for i, color in zip(classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, linestyle='--',
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig("ROCcurve_{}.png".format(name), bbox_inches='tight')

    precision, recall, average_precision, auc_scores = {}, {}, {}, {}
    for n in range(len(classes)):
        precision[n], recall[n], _ = precision_recall_curve(y_test, y_pred)
        average_precision[n] = average_precision_score(y_test, y_pred, average='weighted')
        print('{} Average Precision is {}'.format(n, average_precision[n]))

    lines = []
    labels = []
    plt.figure()
    plt.style.use('ggplot')

    for i, color in zip(classes, colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append("{} AP: {:0.2f}".format(i, average_precision[i]))

    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(lines, labels, title="PR-AUC", loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.savefig("PR_{}.png".format(name), bbox_inches='tight')

    print('Precision Recall Curve generated!')


    return roc_auc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Name of model")
    parser.add_argument('--n', help="name")
    parser.add_argument('--tr', help="filepath and name of data")
    parser.add_argument('--te', help="filepath and name of data")
    parser.add_argument('--ev', help="filepath and name of data")
    args = parser.parse_args()

    train, test, eval = pd.read_csv(args.tr), pd.read_csv(args.te), pd.read_csv(args.ev)
    X_train, Y_train = train['CLEAN_TEXT'], train['bins'].replace({0:0, 1:1, 2:0, 3:0, 4:0})
    X_test, Y_test = test['CLEAN_TEXT'], test['bins'].replace({0:0, 1:1, 2:0, 3:0, 4:0})
    X_eval, Y_eval = eval['CLEAN_TEXT'], eval['bins'].replace({0:0, 1:1, 2:0, 3:0, 4:0})

    words = X_train.append(X_eval)

    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 3),
    norm='l2',
    smooth_idf=False,
    max_features=15000)

    word_vectorizer.fit(words)
    train_word_features = word_vectorizer.transform(X_train)
    eval_word_features = word_vectorizer.transform(X_eval)

    print("Train...")
    lightgbm_params = {
    'boost_from_average': False,
    'objective':'binary',
    'boosting':'gbdt',
    'learning_rate':0.01,
    'min_data_in_leaf':10,
    'verbose': 0,
    'num_leaves': 50,
    'scale_pos_weight': 30.0
    }


    train_word_features = word_vectorizer.transform(words)
    Y_train = Y_train.append(Y_eval)
    lightgbm = trainLightGBM(train_word_features, eval_word_features, Y_train, Y_eval, lightgbm_params)
    print('Saving model...')
    lightgbm.save_model('lightGBM_{}.txt'.format(args.n))

    print("Test results...")
    test_word_features = word_vectorizer.transform(X_test)
    evaluateGBModelsingle(lightgbm, test_word_features, Y_test, [0], args.n)

