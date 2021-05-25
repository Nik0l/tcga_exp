# Libraries ###########################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import random
from sklearn.metrics import roc_curve, roc_auc_score

# Functions ###########################################################################################################
def apply_z_score(df, columns, index):
    # Scale RNAseq data using z-scores
    df = preprocessing.StandardScaler().fit_transform(df)
    df = pd.DataFrame(df, columns=columns, index=index)
    return df

def normalize_data(df, columns, index):
    df = preprocessing.MinMaxScaler().fit_transform(df)
    df = pd.DataFrame(df, columns=columns, index=index)
    return df

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

def prepareScoringData(df_rna, df_mut, genes):
    # Filter df_rna using gene_list
    # Add mutation status as last column
    gene_list = genes
    mutation = df_mut['mutation_status'].tolist()
    df_training = df_rna[df_rna.columns.intersection(gene_list)]
    df_training = df_training.assign(mutation = mutation)
    return(df_training)

def kaufmanScore(df_training):
    # Computes Kaufman score for each sample (arithmetic mean of all gene values)
    ncol = len(df_training.columns)
    score = df_training.iloc[:, 1:ncol-1].apply(np.mean, axis=1)
    df_score = df_training.assign(score = score)
    return(df_score)

def plotScores(df_score, zeroone_norm = True):
    # plots Kaufman scores
    fig_dims = (8, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    plot = sns.violinplot(x='mutation', y='score', data=df_score, palette=["darkblue", "red"])
    plot.set_xlabel('STK11', fontsize=15)
    plot.set_ylabel('K score', fontsize=15)
    sns.despine()
    if zeroone_norm:
        plot.axes.set_title('Kaufman score\n (TMM, Z-score, zero-one counts)', fontsize=20, weight='bold')
    else:
        plot.axes.set_title('Kaufman score\n (TMM, Z-score counts)', fontsize=20, weight='bold')
    return(fig, ax)

def plotROC(fpr, tpr):
    fig_dims = (8, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.title('ROC Kaufman score', fontsize=20, weight=10)
    return(fig, ax)


# Load data ###########################################################################################################
path = 'data/sophie_ML/'
# df_rna: counts normalized by TMM in edgeR
# protein-coding genes only, tumor samples only, outlier plate samples excluded
df_rna = pd.read_csv(path + 'TMM_counts.csv')
# df_mut: info about samples (mutation_status)
df_mut = pd.read_csv(path + 'sample_info_filtered.csv')
# genes: list of gene symbols (protein coding genes only, corresponding to df_rna (order preserved)
genes = pd.read_csv(path + 'genes.csv')
# exact_test: table with exact test results (edgeR) for all protein-coding genes
exact_test = pd.read_csv(path + 'ExactTest_pc_genes.csv')

# Pre-process data ####################################################################################################
# Transpose df_rna (to have genes as columns, samples as rows)
df_rna = df_rna.rename(index = genes['Symbol'])
df_rna_t = df_rna.transpose()

# Standardize data with Z-score
columns = list(df_rna_t.columns)
index = df_rna_t.index
df_rna_zscore = apply_z_score(df_rna_t, columns=columns, index=index)
df_rna_zscore.to_csv('data/sophie_ML/df_rna_TMM_zscore.csv')

# Scale data using zero-one normalization
df_rna_zscore_zeroone = normalize_data(df_rna_zscore, columns=columns, index=index)
df_rna_zscore_zeroone.to_csv('data/sophie_ML/df_rna_TMM_zscore_zeroone.csv')

# Kaufman scoring #####################################################################################################
# get list of genes of interest #######################################################################################
# 16-signature genes
kaufman_genes = ['DUSP4', 'PDE4D', 'IRS2', 'BAG1', 'HAL', 'TACC2', 'AVPI1', 'CPS1', 'PTP4A1', 'RFK',
                 'SIK1', 'FGA','GLCE', 'TESC', 'MUC5AC', 'TFF1']

# Nikolay's list - top10
top10_nik = ['SLC22A6', 'C6orf176', 'COL25A1', 'SLC14A2', 'GLTPD2', 'AGXT2L1', 'CALCA', 'FXYD4', 'C1orf64', 'INHA']

# Ranked gene lists
FDR_genes_top16 = getGeneList(exact_test, 16, rank_by='FDR', logFC_filter=True)
FDR_genes_top100 = getGeneList(exact_test, 100, rank_by='FDR', logFC_filter=True)
FDR_genes_top1000 = getGeneList(exact_test, 1000, rank_by='FDR', logFC_filter=True)
FDR_genes_top5 = getGeneList(exact_test, 5, rank_by='FDR', logFC_filter=True)

logFC_up_genes_top16 = getGeneList(exact_test, 16, rank_by='logFC_up', logFC_filter=True)
logFC_up_genes_top100 = getGeneList(exact_test, 100, rank_by='logFC_up', logFC_filter=True)

logFC_down_genes_top16 = getGeneList(exact_test, 16, rank_by='logFC_down', logFC_filter=True)
logFC_down_genes_top100 = getGeneList(exact_test, 100, rank_by='logFC_down', logFC_filter=True)

# get random gene list
n = random.sample(range(0, len(exact_test.index)), 16) # generate 16 random indexes
random_genes = exact_test.iloc[n, 1]

# Scoring ############################################################################################################
# Prepare data for scoring (select relevant genes and get mutation status)
data = prepareScoringData(df_rna_zscore_zeroone, df_mut, kaufman_genes)

# Score samples and plot data
df_score = kaufmanScore(data)
fig,ax = plotScores(df_score)
fig.savefig('results/violin_kaufman_logFC_down_top100_genes.png')

# ROC curves to evaluate scoring performance
mutation = df_score['mutation'].tolist()
score = df_score['score'].tolist()
fpr, tpr, thresh = roc_curve(mutation, score, pos_label=1)
fig, ax = plotROC(fpr, tpr)
fig.savefig('results/kaufman_ROC_logFC_down_top100_genes.png')
auc_score = roc_auc_score(mutation, score)

# randomized control - shuffle labels
random.shuffle(mutation)
fpr_rand, tpr_rand, thresh_rand = roc_curve(mutation, score, pos_label=1)
fig, ax = plotROC(fpr_rand, tpr_rand)
fig.savefig('results/kaufman_ROC_randomized.png')
auc_score_random = roc_auc_score(mutation, score)

# Misc ###############################################################################################################
# check intersections between gene lists
np.intersect1d(FDR_genes_top16, logFC_up_genes_top16) # ['BLOC1S5-TXNDC5', 'INHA', 'NNAT', 'NPY'] - 4 genes overlap

np.intersect1d(FDR_genes_top100, logFC_up_genes_top100)
# 48 genes overlap
# ['ASPG', 'BAALC', 'BLOC1S5-TXNDC5', 'BMP6', 'BPIFA2', 'BPIFB2',
#  'CALCA', 'CALCB', 'CBR1', 'CCDC154', 'CHGB', 'CHRDL2', 'CRB1',
#  'EYS', 'FAM163A', 'FAM9B', 'FGL1', 'GALNTL6', 'GLTPD2', 'GREB1',
#  'HPX', 'INHA', 'KLRC2', 'LCN15', 'LILRA2', 'LRRC26', 'MAFA',
#  'MARCHF4', 'MTMR7', 'NNAT', 'NPAS3', 'NPY', 'NTNG2', 'ODC1',
#  'PAK3', 'PTPRN', 'RELN', 'RERGL', 'RET', 'SLC16A14', 'SLC26A4',
#  'SLC7A2', 'SPP2', 'SRARP', 'TAF1L', 'TENM1', 'TXNDC2', 'ZACN']

np.intersect1d(FDR_genes_top16, logFC_down_genes_top16) # no overlap

np.intersect1d(FDR_genes_top100, logFC_down_genes_top100)
# 7 genes overlap
# ['ADGRF1', 'ALB', 'FAT2', 'GJA3', 'IVL', 'SHISA3', 'SLC6A20']