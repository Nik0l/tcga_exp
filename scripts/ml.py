import pandas as pd
import matplotlib.pyplot as plt
from xgboost import cv
import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns

from scripts.preprocessing import prepare_data_for_ml


def plot_roc(fpr, tpr, roc_auc):
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
    Y = pd.DataFrame(np.random.randint(0, 2, Y.shape[0]), columns=['STK11'])
    return Y


def get_train_test_data(df_rna, df_gt, column_name, randomise_gt=False):

    print(df_rna.shape, df_gt.shape)
    print(df_rna)
    X = df_rna
    Y = df_gt[column_name]
    print(X, Y)

    if randomise_gt:
        Y = randomised_gt(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=Y)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('total train:', y_train.sum())
    print('total test', y_test.sum())
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    return dtrain, dtest, y_test, y_test


def plot_boxplot(X, Y, column='STK11'):
    df_box = pd.concat([X, Y], axis=1)
    boxplot = df_box.boxplot(by=column)
    plt.show()


def get_params():
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
    xgb_cv = cv(dtrain=dtrain,
                params=get_params(),
                nfold=10,
                num_boost_round=5,
                metrics="auc",
                as_pandas=True,
                seed=42)
    print('Cross validation results: \n', xgb_cv)


def get_genes_from_deg():
    # these are from DEG analysis
    top10 = ['SLC22A6', 'C6orf176', 'COL25A1', 'SLC14A2', 'GLTPD2', 'AGXT2L1', 'CALCA', 'FXYD4', 'C1orf64', 'INHA']
    bottom10 = ['DPCR1', 'MYCN', 'AKR1B15', 'SCIN', 'ECEL1', 'SLC7A2', 'FREM1', 'IL6', 'PPP4R4', 'CHST2']
    return top10, bottom10


def run_ml(path, training_data_filename, training_gt_data_filename, column_name, save_model=False):

    df_rna = pd.read_csv(path + training_data_filename)
    df_rna = df_rna.drop(['SAMPLE_BARCODE'], axis=1)

    what_genes = 'top10'
    top10, bottom10 = get_genes_from_deg()
    if what_genes == 'top10':
        df_rna = df_rna[top10]
    elif what_genes == 'bottom10':
        df_rna = df_rna[bottom10]
    else:
        print('all genes are used')

    df_gt = pd.read_csv(path + training_gt_data_filename)

    dtrain, dtest, y_test, y_test = get_train_test_data(df_rna, df_gt, column_name, randomise_gt=False)
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


path = '~/PycharmProjects/project/tcga_exp/data/' # change to your data path with training data

training_data_filename = 'df_training.csv'
training_gt_data_filename = 'df_gt_training.csv'
cancer_types = ['LUAD']
column_name = 'STK11'

prepare_data_for_ml(path=path,
                    name_df_mut='lung_cancer_mut_only.csv',
                    name_df_rna='lung_cancer_rna_only.csv',
                    name_df_clinical='clinical_data.tsv',
                    training_gt_data_filename=training_gt_data_filename,
                    training_data_filename=training_data_filename,
                    column_name=column_name,
                    cancer_types=cancer_types)

run_ml(path, training_data_filename, training_gt_data_filename, column_name)
