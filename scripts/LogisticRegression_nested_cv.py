# Libraries #########################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import random

# Functions ##########################################################################################################
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

def LogisticRegression_nested_cv(df_rna, df_mut, gene_lists):
    # define the model
    model = LogisticRegression(solver='liblinear',
                               class_weight='balanced',
                               random_state=0)

    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    cv_outer = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    # define hyperparameter search space
    params = {'C': [0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

    # define search
    search = GridSearchCV(model, params, scoring='roc_auc', return_train_score=True, cv=cv_inner, refit=True)

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

def plot_nested_cv_results(scores):
    fig, ax = plt.subplots()
    sns.boxplot(data=scores, palette='Greys')
    sns.despine()
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel('AUROC score', fontsize=12, weight='bold')
    plt.title('Logistic Regression\nnested cross-validation (k=10)', fontsize=15, weight='bold')
    return(fig, ax)

# Load data ##########################################################################################################
path = 'data/sophie_ML/'
df_rna = pd.read_csv(path + 'df_rna_TMM_zscore_zeroone.csv') # df_rna: pre-processed counts table
df_mut = pd.read_csv(path + 'sample_info_filtered.csv') # df_mut: info about samples (mutation_status)
exact_test = pd.read_csv(path + 'ExactTest_pc_genes.csv') # exact_test: table with exact test results (edgeR) for all protein-coding genes

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
n = random.sample(range(0, len(exact_test.index)), 16) # generate 16 random indexes
random_genes_1 = exact_test.iloc[n, :] # to check gene statistics (FDR, logFC...)
random_genes_1_list = exact_test.iloc[n, 1]

n = random.sample(range(0, len(exact_test.index)), 16)
random_genes_2 = exact_test.iloc[n, :]
random_genes_2_list = exact_test.iloc[n, 1]

n = random.sample(range(0, len(exact_test.index)), 16)
random_genes_3 = exact_test.iloc[n, :]
random_genes_3_list = exact_test.iloc[n, 1]

# Logistic regression ###############################################################################################

# prepare lists of genes
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
              'random_genes_1':random_genes_1_list,
              'random_genes_2':random_genes_2_list,
              'random_genes_3':random_genes_3_list}

# perform nested cross-validation to evaluate performance
LR_results, average_scores = LogisticRegression_nested_cv(df_rna, df_mut, gene_lists)

# plot results
fig, ax = plot_nested_cv_results(LR_results)
fig.savefig('results/logistic_regression/LR_nested_cv.png', bbox_inches='tight')