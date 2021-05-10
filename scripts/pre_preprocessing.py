import os
import pandas as pd
from sklearn import preprocessing


def choose_n_most_expressed_genes(df, num_mad_genes=5000):
    # Determine most variably expressed genes and subset
    mad_genes = df.mad(axis=0).sort_values(ascending=False)
    top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
    df = df.loc[:, top_mad_genes]
    return df


def apply_z_score(df, columns, index):
    # Scale RNAseq data using z-scores
    df = preprocessing.StandardScaler().fit_transform(df)
    df = pd.DataFrame(df, columns=columns, index=index)
    return df


def normalize_data(df, columns, index):
    df = preprocessing.MinMaxScaler().fit_transform(df)
    df = pd.DataFrame(df, columns=columns, index=index)
    return df


def process_rnaseq_data(rna_file, num_mad_genes, drop, save, rna_out_file, rna_out_zeroone_file):
    # Processing RNAseq data by z-score and zeroone norm
    # Process RNAseq file
    rnaseq_df = pd.read_table(rna_file, index_col=0)
    rnaseq_df.index = rnaseq_df.index.map(lambda x: x.split('|')[0])
    rnaseq_df.columns = rnaseq_df.columns.str.slice(start=0, stop=15)
    rnaseq_df = rnaseq_df.drop('?').fillna(0).sort_index(axis=1)

    # if gene is listed twice in RNAseq data, drop both occurrences
    if drop != '':
        rnaseq_df.drop(drop, axis=0, inplace=True)
    rnaseq_df = rnaseq_df.T

    rnaseq_subset_df = choose_n_most_expressed_genes(rnaseq_df, num_mad_genes=num_mad_genes)
    rnaseq_scaled_df = apply_z_score(rnaseq_subset_df, columns=rnaseq_subset_df.columns, index=rnaseq_subset_df.index)
    if save:
        rnaseq_scaled_df.to_csv(rna_out_file, sep='\t', compression='gzip')
    # Scale RNAseq data using zero-one normalization
    rnaseq_scaled_zeroone_df = normalize_data(rnaseq_subset_df, columns=rnaseq_subset_df.columns, index=rnaseq_subset_df.index)
    if save:
        rnaseq_scaled_zeroone_df.to_csv(rna_out_zeroone_file, sep='\t', compression='gzip')


def process_mutations(mutation_df, mut_out_file, save):
    # Process mutation in long format to dataframe format
    mut_pivot = (mutation_df.query("effect in @mutations")
                 .groupby(['#sample', 'chr', 'gene'])
                 .apply(len).reset_index()
                 .rename(columns={0: 'mutation'}))

    mut_pivot = (mut_pivot.pivot_table(index='#sample',
                                       columns='gene',
                                       values='mutation',
                                       fill_value=0)
                          .astype(bool).astype(int))
    mut_pivot.index.name = 'SAMPLE_BARCODE'
    if save:
        mut_pivot.to_csv(mut_out_file, sep='\t', compression='gzip')
