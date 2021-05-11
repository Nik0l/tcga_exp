import pandas as pd
import matplotlib.pyplot as plt
from xgboost import cv
import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns


def plot_roc(fpr, tpr, roc_auc):
    """ Plot ROC curve. """
    #fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (area = %0.6f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.show()


def randomised_gt(Y):
    """ Get a random Y as a sanity check. """
    Y = pd.DataFrame(np.random.randint(0, 2, Y.shape[0]), columns=['STK11'])
    return Y


def get_train_test_data(X, df_gt, column_name, test_size, randomise_gt=False):
    """ Split the data into training and test"""

    Y = df_gt[column_name]
    if randomise_gt:
        Y = randomised_gt(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=Y)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('total train samples:', y_train.sum())
    print('total test samples', y_test.sum())
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    return dtrain, dtest, y_test, y_test


def get_params():
    """ All of xgboost parameters for training. """
    params = {
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'verbose': 1,
        'max_depth': 6,
        'min_child_weight': 4,
        'gamma': 0.6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 5e-05,
        'max_depth': 10,
        'objective': 'binary:logistic',
        'nthread': 20,
        # 'scale_pos_weight': w,
        'seed': 42}
    return params


def plot_corr(df_rna, df_gt, column_name):
    """ Plot correlation matrices. """
    rs = np.random.RandomState(0)
    df = pd.concat([df_rna, df_gt[column_name]], axis=1)
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')
    print(corr)
    sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plt.show()


def run_cv(dtrain):
    """ Run cross validaiton. Important: make sure that your model does not overfit."""
    xgb_cv = cv(dtrain=dtrain,
                params=get_params(),
                nfold=10,
                num_boost_round=5,
                metrics="auc",
                as_pandas=True,
                seed=42)
    print('Cross validation results: \n', xgb_cv)


def run_ml(path, training_data_filename, training_gt_data_filename, column_name, genes_subset=None, test_size=0.2, save_model=False):
    """ Main function to train an xgboost classifier, save it, evaluate, plot importance of its features."""
    df_rna = pd.read_csv(path + training_data_filename)
    df_rna = df_rna.drop(['SAMPLE_BARCODE'], axis=1)

    if genes_subset is not None:
        df_rna = df_rna[genes_subset]
    else:
        print('all genes are used!')

    df_gt = pd.read_csv(path + training_gt_data_filename)

    dtrain, dtest, y_test, y_test = get_train_test_data(df_rna, df_gt, column_name,
                                                        test_size=test_size,
                                                        randomise_gt=False)
    bst = xgb.train(params=get_params(), dtrain=dtrain, num_boost_round=100)

    if save_model:
        bst.dump_model('dump.raw.txt')

    cv_on = True
    if cv_on:
        run_cv(dtrain)

    preds = bst.predict(dtest)
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, roc_auc)

    #print(precision_score(y_test, best_preds, average='macro'))
    #print(recall_score(y_test, best_preds, average='macro'))

    s = bst.get_score(importance_type='gain')
    importance = pd.DataFrame.from_dict(s, orient='index')
    importance.columns = ['score']
    importance = importance.sort_values(by='score', ascending=False)
    print(importance)

    if save_model:
        joblib.dump(bst, 'bst_model.pkl', compress=True)
