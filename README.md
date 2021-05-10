# tcga_exp
TCGA RNAseq and H&amp;E WSIs

0. Pre-pre-process data similar to 'Tybalt' https://github.com/greenelab/tybalt;

1. Pre-processing the data for ML classification, for example, STK11 mutation prediction;

2. Running Differential Gene Expression (DEG) analysis to identify genes;

3. Training a ML model for classification.


Run `exp.py`:

```python
# preprocess the data
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

# see top 10 DEG genes
top_10_genes = get_10top(path + path_to_deg)
print(top_10_genes)

# train models
run_ml(path, training_data_filename, training_gt_data_filename, column_name)

```
