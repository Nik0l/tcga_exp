import pandas as pd
import matplotlib.pyplot as plt
from scripts.pre_preprocessing import process_mutations, process_rnaseq_data


def pre_preprocessing(rna_file='', mut_file = '', rna_out_file=''):
    # IMPORTANT: for reference only, no need in using the function
    ''' Pre-preprocessing generates two files what are already added in 'data:
                    name_df_mut='lung_cancer_mut_only.csv',
                    name_df_rna='lung_cancer_rna_only.csv',
    '''
    rna_out_zeroone_file = 'lung_cancer_rna.csv' # filter only LUAD, if necessary
    mut_out_file = 'lung_cancer_mutations.csv'  # filter only LUAD, if necessary
    process_rnaseq_data(rna_file, num_mad_genes=5000, drop='SLC35E2', save=True,
                        rna_out_file=rna_out_file, rna_out_zeroone_file=rna_out_zeroone_file)
    mutation_df = pd.read_table(mut_file)
    process_mutations(mutation_df, mut_out_file, save=True)


def genes_lists_to_play_with():
    # these are from DEG analysis
    gene_list = ['SLC22A6', 'C6orf176', 'COL25A1', 'SLC14A2', 'GLTPD2', 'AGXT2L1', 'CALCA', 'FXYD4', 'C1orf64', 'INHA']
    #bottom10 = ['DPCR1', 'MYCN', 'AKR1B15', 'SCIN', 'ECEL1', 'SLC7A2', 'FREM1', 'IL6', 'PPP4R4', 'CHST2']
    return gene_list


def plot_boxplot(X, Y, column='STK11'):
    df_box = pd.concat([X, Y], axis=1)
    boxplot = df_box.boxplot(by=column)
    plt.show()