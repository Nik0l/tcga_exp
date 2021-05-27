import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate

# Functions #########################################################################################################
def getGeneList(df_tested_genes, n, rank_by='FDR', logFC_filter=True):
    '''
    @param df_tested_genes: dataframe containing gene symbols and their respective FDR and logFC values
    @param n: number of genes to be returned
    @param rank_by: variable to be used for ranking. Possible values: 'FDR', 'logFC_up', 'logFC_down'
    @param logFC_filter: if True, keeps only genes with (logFC ≤ -1 | logFC ≥ 1)
    '''
    if logFC_filter:
        genes = df_tested_genes.loc[(df_tested_genes['logFC'] < -1) | (df_tested_genes['logFC'] > 1)]

    if rank_by == 'logFC_up':
        genes = df_tested_genes[df_tested_genes['FDR'] <= 0.01]  # keep significant genes only
        genes = genes.sort_values(by='logFC', ascending=False)
    elif rank_by == 'logFC_down':
        genes = df_tested_genes[df_tested_genes['FDR'] <= 0.01]  # keep significant genes only
        genes = genes.sort_values(by='logFC', ascending=True)
    else:
        genes = genes.sort_values(by='FDR')
    gene_list = genes.iloc[:n, 1].tolist()
    return(gene_list)

def getRandomGeneList(df_tested_genes, n):
    '''
    Returns a random gene list, as well as a dataframe containing statistics for the selected genes
    @param df_tested_genes: dataframe containing gene symbols and their respective FDR and logFC values
    @param n: number of genes to be returned
    '''
    indexes = random.sample(range(0, len(df_tested_genes.index)), n) # generate n random indexes
    random_df = df_tested_genes.iloc[indexes, :]
    random_list = df_tested_genes.iloc[indexes, 1]
    return(random_df, random_list)

def prepareData(df_rna, df_mut, gene_list):
    '''
    Filters df_rna using gene_list and creates X (input variables) and y (labels) np arrays
    @param df_rna: dataframe with normalized and standardized gene counts for all samples
    @oaram df_mut: dataframe containing sample barcodes and their corresponding STK11 mutation status
                   sample order must be the same as for df_rna
    @param gene_list: list of the gene symbols of interest
    '''
    y = df_mut['mutation_status'].values
    X = df_rna[df_rna.columns.intersection(gene_list)].values
    return(X, y)

def nestedCV(model, params, df_rna, df_mut, gene_lists):
    '''
    Performs nested cross-validation to optimise model hyperparameters and evaluate performance
    @param model: the model to be trained
    @param params: hyperparameter the search space
    @param: df_rna: dataframe with normalized and standardized gene counts for all samples
    @param df_mut: dataframe containing sample barcodes and their corresponding STK11 mutation status
                   -- sample order must be the same as for df_rna
    @param gene_lists: dictionary containing the gene_list names and corresponding gene symbols
    Returns cv_results: dictionary with full cross-validation results for each gene_list (as keys)
    Returns df_scores: dataframe with test scores for all outer folds
    Returns average_scores: dataframe with mean and std of scores for all gene_lists
    '''
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    cv_outer = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    # define search
    search = GridSearchCV(model, params,
                          scoring='roc_auc',
                          return_train_score=True,
                          cv=cv_inner,
                          refit=True,
                          n_jobs=-1)

    # create empty dictionary and dataframe to store results
    cv_results = dict.fromkeys(gene_lists.keys(), None)
    df_scores = pd.DataFrame(columns=gene_lists.keys())

    for i, k in enumerate(gene_lists.keys()):
        # prepare X and y using the list of genes of interest
        X, y = prepareData(df_rna, df_mut, gene_lists[k])

        # execute the nested cross-validation and get results
        cv_out = cross_validate(search, X, y,
                                scoring='roc_auc',
                                cv=cv_outer,
                                n_jobs=-1,
                                return_train_score=True,
                                return_estimator=True)
        cv_results[k] = cv_out
        df_scores[k] = cv_out['test_score']
        print('gene list ' + str(i + 1) + ' of ' + str(len(gene_lists.keys())))

    # compute average scores and std
    average_scores = pd.DataFrame(index=gene_lists.keys(), columns=['average_roc_auc', 'std'])
    average_scores['average_roc_auc'] = df_scores.apply(np.mean, axis='index')
    average_scores['std'] = df_scores.apply(np.std, axis='index')

    return (cv_results, df_scores, average_scores)

def plot_nested_cv_results(scores, y_lim=None, title=None):
    fig, ax = plt.subplots()
    sns.boxplot(data=scores, palette='Greys')
    sns.despine()
    plt.ylim(y_lim)
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel('AUROC score', fontsize=12, weight='bold')
    plt.title(title, fontsize=15, weight='bold')
    return(fig, ax)

def getEstimatorParams(cv_results, name, params):
    '''
    Returns dataframe with hyperparameters for the best estimators after cross-validation
    @param cv_resuls: output from sklearn.cross_validate(return_estimator=True)
    @param name: name of the gene_list of interest (must be a gene_lists key)
    @param params: list containing the names of the hyperparameters that were optimised. e.g. ['n_estimators', 'min_samples_split', 'max_depth']
    '''
    k = len(cv_results[name]['estimator'])  # find number of outer K-folds
    param_df = pd.DataFrame(columns=range(len(params)))
    for i in range(k):
        all_params = cv_results[name]['estimator'][i].best_estimator_.get_params()
        values = [all_params[x] for x in params]
        param_df = param_df.append(np.reshape(values, (1, -1)).tolist())
    param_df.columns = params
    return (param_df)

def getFeatureImportance(cv_results, name, gene_lists, model_type):
    '''
    Creates dataframe with feature importance from the best estimators for each outer fold
    @param cv_results: output from sklearn.cross_validate(return_estimator=True)
    @param name: name of the gene_list of interest (must be a gene_lists key)
    @param gene_lists: dictionary containing the gene_list names and corresponding gene symbols
    @param model_type: 'LR' for Logistic regression or 'RF' for Random Forest
    Returns a dataframe containing feature importance values for each gene (columns) and runs (indexes)
    '''
    k = len(cv_results[name]['estimator'])  # find number of outer K-folds
    n = len(gene_lists[name])  # find number of genes
    feat_df = pd.DataFrame(columns=range(n))

    if model_type == 'RF':
        for i in range(k):
            features = cv_results[name]['estimator'][i].best_estimator_.feature_importances_
            feat_df = feat_df.append(np.reshape(features, (1, -1)).tolist())
        feat_df.columns = gene_lists[name]
    elif model_type == 'LR':
        for i in range(k):
            features = cv_results[name]['estimator'][i].best_estimator_.coef_
            feat_df = feat_df.append(features.tolist())
        feat_df.columns = gene_lists[name]

    return (feat_df)

def plotFeatureImportance(feat_df, y_lim=None, title=None):
    fig, ax = plt.subplots()
    sns.boxplot(data=feat_df, palette='Greys')
    sns.despine()
    plt.ylim(y_lim)
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel('Values', fontsize=12, weight='bold')
    plt.title(title, fontsize=15, weight='bold')
    return(fig, ax)

# Load data ##########################################################################################################
path = 'data/sophie_ML/'
df_rna = pd.read_csv(path + 'df_rna_TMM_zscore_zeroone.csv') # pre-processed counts table
df_mut = pd.read_csv(path + 'sample_info_filtered.csv') # dataframe with info about samples (mutation_status)
exact_test = pd.read_csv(path + 'ExactTest_pc_genes.csv') # dataframe with exact test results (edgeR) for all protein-coding genes

# Make gene lists ####################################################################################################
# 16-signature genes
kaufman_genes = ['DUSP4', 'PDE4D', 'IRS2', 'BAG1', 'HAL', 'TACC2', 'AVPI1', 'CPS1', 'PTP4A1', 'RFK',
                 'SIK1', 'FGA','GLCE', 'TESC', 'MUC5AC', 'TFF1']

# Ranked gene lists
FDR_genes_top16 = getGeneList(exact_test, 16, rank_by='FDR', logFC_filter=True)
FDR_genes_top100 = getGeneList(exact_test, 100, rank_by='FDR', logFC_filter=True)
FDR_genes_top1000 = getGeneList(exact_test, 1000, rank_by='FDR', logFC_filter=True)
FDR_genes_top5 = getGeneList(exact_test, 5, rank_by='FDR', logFC_filter=True)
FDR_genes_top50 = getGeneList(exact_test, 50, rank_by='FDR', logFC_filter=True)
logFC_up_genes_top5 = getGeneList(exact_test, 5, rank_by='logFC_up', logFC_filter=True)
logFC_up_genes_top16 = getGeneList(exact_test, 16, rank_by='logFC_up', logFC_filter=True)
logFC_up_genes_top50 = getGeneList(exact_test, 50, rank_by='logFC_up', logFC_filter=True)
logFC_up_genes_top100 = getGeneList(exact_test, 100, rank_by='logFC_up', logFC_filter=True)
logFC_up_genes_top1000 = getGeneList(exact_test, 1000, rank_by='logFC_up', logFC_filter=True)

# Non-significant genes
FDR_genes_bottom16 = exact_test.sort_values(by='FDR', ascending=False)
FDR_genes_bottom16 = FDR_genes_bottom16.iloc[:16, 1].tolist()
FDR_genes_bottom100 = exact_test.sort_values(by='FDR', ascending=False)
FDR_genes_bottom100 = FDR_genes_bottom100.iloc[:100, 1].tolist()

# Random gene sets
random_df_1, random_list_1 = getRandomGeneList(exact_test, 16)
random_df_2, random_list_2 = getRandomGeneList(exact_test, 16)
random_df_3, random_list_3 = getRandomGeneList(exact_test, 16)

# Genes selected based on RF feature importance
top_5_important = ['NPY', 'EYS', 'MTMR7', 'ODC1', 'SLC16A14']
top_17_important = ['NPY', 'EYS', 'MTMR7', 'ODC1', 'SLC16A14', 'NNAT', 'ZACN', 'CACNB2', 'INHA', 'GREB1', 'HPX',
                    'FURIN', 'TENM1', 'ASPG', 'RPH3AL', 'RET', 'CRB1']

# prepare gene_lists object
gene_lists = {'k_genes':kaufman_genes,
              'top_5_FDR':FDR_genes_top5,
              'top_16_FDR':FDR_genes_top16,
              'top_50_FDR':FDR_genes_top50,
              'top_100_FDR':FDR_genes_top100,
              'top_1000_FDR':FDR_genes_top1000,
              'top_5_logFC':logFC_up_genes_top5,
              'top_16_logFC':logFC_up_genes_top16,
              'top_50_logFC':logFC_up_genes_top50,
              'top_100_logFC':logFC_up_genes_top100,
              'top_1000_logFC':logFC_up_genes_top1000,
              'bottom_16_FDR':FDR_genes_bottom16,
              'bottom_100_FDR':FDR_genes_bottom100,
              'random_genes_1':random_list_1,
              'random_genes_2':random_list_2,
              'random_genes_3':random_list_3}

important_gene_list = {'top_5_FDR':FDR_genes_top5,
                       'top_5_important': top_5_important,
                       'top_17_important': top_17_important,
                       'top_16_FDR':FDR_genes_top16,
                       'top_50_FDR':FDR_genes_top50}

# Logistic regression ##############################################################################################
# define the model
LR_model = LogisticRegression(solver='liblinear',
                              class_weight='balanced',
                              random_state=0)

# define hyperparameter search space
LR_params = {'C': [0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2']}

# run nested CV
LR_cv_results, LR_cv_scores, LR_cv_average_scores = nestedCV(LR_model, LR_params, df_rna, df_mut, important_gene_list)

# plot cv scores
fig, ax = plot_nested_cv_results(LR_cv_scores,
                                 y_lim=[0.2, 1.1],
                                 title='Logistic Regression\nnested cross-validation (k=10)')
fig.savefig('results/logistic_regression/LR_nested_cv_important_genes.png', bbox_inches='tight')

# get best estimators parameters for each outer run
param_df = getEstimatorParams(LR_cv_results,
                              name='top_50_FDR',
                              params=['C', 'penalty'])

# Get coefficients of best estimators for all outer runs
coef_df = getFeatureImportance(LR_cv_results, name='top_16_FDR', gene_lists=gene_lists, model_type='LR')

# Plot results
fig, ax = plotFeatureImportance(coef_df, y_lim=[-5,20], title='LR coefficients\ntop 16 FDR genes')
fig.savefig('results/logistic_regression/LR_coefficients_top_16_FDR.png', bbox_inches='tight')


# Random Forest ##############################################################################################
# define the model
RF_model = RandomForestClassifier(class_weight='balanced',
                                  random_state=0)

# define hyperparameter search space
RF_params = {'n_estimators': [50, 100],
             'max_depth': [2, 3, 4]}

# run nested CV
RF_cv_results, RF_cv_scores, RF_cv_average_scores = nestedCV(RF_model, RF_params, df_rna, df_mut, important_gene_list)

# plot cv scores
fig, ax = plot_nested_cv_results(RF_cv_scores,
                                 y_lim=[0.2, 1.1],
                                 title='Random Forest\nnested cross-validation (k=10)')
fig.savefig('results/random_forest/RF_nested_cv_important_genes.png', bbox_inches='tight')

# get best estimators hyperparameters for each outer run
RF_param_df = getEstimatorParams(RF_cv_results,
                                 name='top_50_FDR',
                                 params=['n_estimators', 'max_depth'])

# Get feature importances of best estimators for all outer runs
feat_df = getFeatureImportance(RF_cv_results, name='top_50_FDR', gene_lists=gene_lists, model_type='RF')

# Plot results
fig, ax = plotFeatureImportance(feat_df, y_lim=None, title='RF feature importance\ntop 50 FDR genes')
fig.savefig('results/random_forest/RF_feature_importance_top_50_FDR.png', bbox_inches='tight')