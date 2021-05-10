from scripts.preprocessing import prepare_data_for_ml
from scripts.ml import run_ml
from scripts.deg import run_deg
from scripts.deg import get_10top


def pre_preprocessing():
    # IMPORTANT: for reference only, no need in using the function
    ''' Pre-preprocessing generates two files what are already added in 'data:
                    name_df_mut='lung_cancer_mut_only.csv',
                    name_df_rna='lung_cancer_rna_only.csv',
    '''
    import pandas as pd
    from scripts.pre_preprocessing import process_mutations, process_rnaseq_data

    rna_file = ''
    rna_out_file = ''
    rna_out_zeroone_file = 'lung_cancer_rna.csv' # filter only LUAD, if necessary
    mut_file = ''
    mut_out_file = 'lung_cancer_mutations.csv'  # filter only LUAD, if necessary
    process_rnaseq_data(rna_file, num_mad_genes=5000, drop='SLC35E2', save=True,
                        rna_out_file=rna_out_file, rna_out_zeroone_file=rna_out_zeroone_file)
    mutation_df = pd.read_table(mut_file)
    process_mutations(mutation_df, mut_out_file, save=True)


path = '~/PycharmProjects/project/tcga_exp/data/' # IMPORTANT: change to your data path with the training data

training_data_filename = 'df_training.csv'
training_gt_data_filename = 'df_gt_training.csv'
cancer_types = ['LUAD']
column_name = 'STK11'
path_to_deg = 'deg.csv'


#pre_preprocessing()

# prepare the data
prepare_data_for_ml(path=path,
                    name_df_mut='lung_cancer_mut_only.csv',
                    name_df_rna='lung_cancer_rna_only.csv',
                    name_df_clinical='clinical_data.tsv',
                    training_data_filename=training_data_filename,
                    training_gt_data_filename=training_gt_data_filename,
                    column_name=column_name,
                    cancer_types=cancer_types)

# run DEG analysis
run_deg(column_name=column_name,
        path=path,
        training_data_filename=training_data_filename,
        training_gt_data_filename=training_gt_data_filename,
        output_results_file=path_to_deg)

top_10_genes = get_10top(path + path_to_deg)
print(top_10_genes)

# train models
run_ml(path, training_data_filename, training_gt_data_filename, column_name)