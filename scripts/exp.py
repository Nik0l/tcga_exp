from scripts.preprocessing import prepare_data_for_ml
from scripts.ml import run_ml
from scripts.deg import run_deg
from scripts.deg import get_10top
from scripts.deg import get_10bottom


def get_subset_genes(path_to_deg, what_genes='top10'):
    """ Get a list of genes: 'top10', 'bottom10', 'all' otherwise. """
    if what_genes == 'top10':
        genes_subset = get_10top(path_to_deg)
    elif what_genes == 'bottom10':
        genes_subset = get_10bottom(path_to_deg)
    else:
        genes_subset = None
        print('all genes are used')
    return genes_subset


path = '~/PycharmProjects/project/tcga_exp/data/' # IMPORTANT: change to your data path with the training data
training_data_filename = 'df_training.csv'
training_gt_data_filename = 'df_gt_training.csv'
path_to_deg = 'deg.csv'

cancer_types = ['LUAD']
column_name = 'STK11'

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
        output_results_file=path+path_to_deg)
# get top10, bottom10 or all genes from DEG; if None then all genes are used
genes_subset = get_subset_genes(path + path_to_deg, what_genes='top10')
# train models
run_ml(path, training_data_filename, training_gt_data_filename, column_name, genes_subset=genes_subset)
