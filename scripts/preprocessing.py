import pandas as pd


def get_clinical_data(path_clinical, cancer_types):
    """ Some preprocessing for TCGA clinical data."""
    df_clinical = pd.read_csv(path_clinical, sep='\t')
    df_clinical = df_clinical[df_clinical['acronym'].isin(cancer_types)]
    df_clinical = df_clinical.rename(columns={'acronym': 'cancer_type'})
    df_clinical['SAMPLE_BARCODE'] = df_clinical['portion_id'].str[:-4]
    if len(cancer_types) > 1:
        df_clinical['type'] = df_clinical['cancer_type'] + '_' + df_clinical['sample_type']
        df_clinical = df_clinical[['SAMPLE_BARCODE', 'sample_type', 'type']]
    else:
        df_clinical = df_clinical[['SAMPLE_BARCODE', 'sample_type']]
    return df_clinical


def prepare_data_for_ml(path='~/PycharmProjects/project/tcga_exp/data/',
                        name_df_mut='lung_cancer_mut_only.csv',
                        name_df_rna='lung_cancer_rna_only.csv',
                        name_df_clinical='clinical_data.tsv',
                        training_gt_data_filename='df_gt_training.csv',
                        training_data_filename='df_training.csv',
                        column_name='STK11',
                        cancer_types=['LUAD']):
    '''
    :param path: path to your data folder
    :param name_df_mut: the file with mutations (SAMPLE_BARCODE, MUT1, ..., MUT_n; mutations are 0 or 1
    :param name_df_rna: 5000 most expressed genes
    :param name_df_clinical: clinical data with normal/tumour data
    :param training_gt_data_filename: where to save the GT data
    :param training_data_filename: where to save the training data
    :param column_name: STK11
    :param cancer_types: [LUAD]
    :return: two files for training with GT
    '''
    df_clinical = get_clinical_data(path + name_df_clinical, cancer_types)
    luad_patients = df_clinical[df_clinical['sample_type'] != 'Solid Tissue Normal']
    luad_patients = list(luad_patients['SAMPLE_BARCODE'])

    df_rna = pd.read_csv(path + name_df_rna)
    df_rna = df_rna[df_rna['SAMPLE_BARCODE'].isin(luad_patients)]

    df_mut = pd.read_csv(path + name_df_mut)
    df_mut = df_mut[['SAMPLE_BARCODE', column_name]]
    df_mut = df_mut[df_mut['SAMPLE_BARCODE'].isin(luad_patients)]

    df_gt = pd.merge(df_rna['SAMPLE_BARCODE'], df_mut.drop_duplicates(),
                     how='left', on='SAMPLE_BARCODE')[['SAMPLE_BARCODE', column_name]]
    df_gt = df_gt.fillna(0.0)

    df_gt.to_csv(path + training_gt_data_filename, index=None)
    df_rna.to_csv(path + training_data_filename, index=None)