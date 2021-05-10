import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import pandas as pd
import scipy.stats
import diffxpy.api as de


def run_deg(column_name='STK11',
            path='~/PycharmProjects/project/tcga_exp/',
            training_data_filename ='df_training.csv',
            training_gt_data_filename='df_gt_training.csv',
            output_results_file='deg.csv',
            save_results=True):

    df_rna = pd.read_csv(path + training_data_filename)
    df_rna = df_rna.drop(['SAMPLE_BARCODE'], axis=1)
    df_gt = pd.read_csv(path + training_gt_data_filename)

    X = df_rna
    Y = df_gt[column_name]

    x = X.to_numpy()
    var_y = X.columns
    Y.columns = ['condition']
    print(x.shape)
    print(var_y)

    print('LUAD data')
    Y = Y.to_frame()
    Y[column_name] = Y[column_name].astype(int)
    Y['batch'] = 0
    print(X.to_numpy())
    print(pd.DataFrame(index=X.columns))
    print(pd.DataFrame(Y, columns=[column_name, 'batch']))

    data = anndata.AnnData(
        X=X.to_numpy(),
        var=pd.DataFrame(index=X.columns),
        obs=pd.DataFrame(Y, columns=[column_name, 'batch'])
    )

    test = de.test.wald(
        data=data,
        formula_loc='~ 1 + ' + column_name,
        factor_loc_totest=column_name
    )
    select_top_n_genes = 10
    print(test.pval[:select_top_n_genes])
    print(test.qval[:select_top_n_genes])
    print(test.summary().iloc[:select_top_n_genes, :])
    df_res = pd.DataFrame(test.summary())
    print(df_res)

    if save_results:
        df_res.to_csv(output_results_file)

    test.plot_volcano(corrected_pval=True, min_fc=1.05, alpha=0.05, size=20)
    test.plot_vs_ttest()


def get_10top(path_to_deg):
    df_res = pd.read_csv(path_to_deg)
    df_res = df_res[['gene', 'pval', 'qval', 'log2fc', 'mean', 'zero_mean', 'grad', 'coef_mle', 'coef_sd', 'll']]
    df_res = df_res.sort_values(by='qval', ascending=True)
    print(df_res.columns)
    top_10_genes = list(df_res.gene[-10:])
    print(df_res.head(10))
    return top_10_genes