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

def prepareTrainingData(df_rna, df_mut, genes):
    # Filter df_rna using gene_list
    # Add mutation status as last column
    gene_list = genes['Symbol'].tolist()
    mutation = df_mut['mutation_status'].tolist()
    df_training = df_rna[gene_list]
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
    plot = sns.stripplot(x='mutation', y='score', data=df_score, palette=["darkblue", "red"])
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
# kaufman_genes: list of genes in 16-gene signature by Kaufman et al., 2014
kaufman_genes = pd.read_csv(path + 'ET_kaufman_genes.csv')

# Pre-process data ####################################################################################################
# Transpose df_rna (to have genes as columns, samples as rows)
df_rna = df_rna.rename(index = genes['Symbol'])
df_rna_t = df_rna.transpose()

# Standardize data with Z-score
columns = list(df_rna_t.columns)
index = df_rna_t.index
df_rna_zscore = apply_z_score(df_rna_t, columns=columns, index=index)
df_rna_zscore.to_csv('data/sophie_ML/df_rna_TMM_zscore', sep='\t', compression='gzip')

# Scale data using zero-one normalization
df_rna_zscore_zeroone = normalize_data(df_rna_zscore, columns=columns, index=index)
df_rna_zscore_zeroone.to_csv('data/sophie_ML/df_rna_TMM_zscore_zeroone', sep='\t', compression='gzip')

# Kaufman scoring #####################################################################################################
# Prepare data for scoring (select relevant genes and get mutation status)
df_training = prepareTrainingData(df_rna_zscore_zeroone, df_mut, kaufman_genes)

# Score samples and plot data
df_score = kaufmanScore(df_training)
fig,ax = plotScores(df_score)
fig.savefig('results/kaufman_score.png')

# ROC curves to evaluate scoring performance
mutation = df_score['mutation'].tolist()
score = df_score['score'].tolist()
fpr, tpr, thresh = roc_curve(mutation, score, pos_label=1)
fig, ax = plotROC(fpr, tpr)
fig.savefig('results/kaufman_ROC.png')
auc_score = roc_auc_score(mutation, score)

# randomized control - shuffle labels
random.shuffle(mutation)
fpr_rand, tpr_rand, thresh_rand = roc_curve(mutation, score, pos_label=1)
fig, ax = plotROC(fpr_rand, tpr_rand)
fig.savefig('results/kaufman_ROC_randomized.png')
auc_score_random = roc_auc_score(mutation, score)
