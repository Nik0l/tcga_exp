import pandas as pd
import numpy as np
import seaborn as sns
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Functions #########################################################################################################
def processEmbeddings(path, file_names):
    '''
    Returns dataframe with mean or median embeddings for each sample (could add other transformations)
    @param path: path to image embeddings
    @param file_names: list of the names of the files to be processed
    '''
    # create empty dataframes to store data
    mean_embeddings = pd.DataFrame(columns = range(512))
    median_embeddings = pd.DataFrame(columns = range(512))
    for i, f in enumerate(file_names):
        embedding = pd.read_csv(path + f + '.csv')
        mean = embedding.apply(np.mean, axis=0).tolist()
        median = embedding.apply(np.median, axis=0).tolist()
        mean_embeddings = mean_embeddings.append(np.reshape(mean, (1, -1)).tolist())
        median_embeddings = median_embeddings.append(np.reshape(median, (1, -1)).tolist())
        print('sample ' + str(i+1) + ' of ' + str(len(file_names)))
    mean_embeddings.index = file_names
    median_embeddings.index = file_names
    return(mean_embeddings, median_embeddings)

def dimReduction(samples, mutation_status, reduction):
    '''
    Returns dataframe with data reduced to 2D and corresponding mutation_status
    @param samples: dataframe containing one embedding per sample (e.g. mean_embeddings)
    @param mutation_status: list of mutation status for all samples in the samples dataframe (must have corresponding order)
    @param reduction: the dimensionality reduction method. Possible values: 'PCA', 'TSNE', 'UMAP'
    '''
    if reduction == 'TSNE':
        reducer = TSNE(n_components=2)
    elif reduction == 'PCA':
        reducer = PCA(n_components=2)
    elif reduction == 'UMAP':
        reducer = UMAP(n_components=2)
    map = reducer.fit_transform(samples.iloc[:,:512])
    df = pd.DataFrame(map, columns=['x', 'y'])
    df['mutation'] = mutation_status
    return(df)

def plotMAP(df, title=None):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='x', y='y', hue='mutation', palette=['darkblue', 'red'], s=30)
    plt.legend(loc='upper left', title='STK11 mutation')
    plt.title(title)
    return(fig, ax)

# Load data #########################################################################################################
path_to_embeddings = 'data/tcga-dataset/tcga_lung_data_feats/'

patients = pd.read_csv('data/sophie_ML/sample_info_filtered.csv') # dataframe containing list of patients to be used
patient_list = patients['cases.submitter_id'].unique() # gets list of patient IDs

wsi_list = pd.read_csv('data/tcga-dataset/LUAD.csv')
wsi_list['cases.submitter_id'] = wsi_list['0'].str[20:32] # adds column with patient ID
wsi_list['file_name'] = wsi_list['0'].str[20:] # adds column with embeddings file names
wsi_filtered = wsi_list.loc[wsi_list['cases.submitter_id'].isin(patient_list)] # keep only patients in patients_list
wsi_filtered = pd.merge(wsi_filtered, patients[['cases.submitter_id', 'mutation_status']].drop_duplicates(),
                        how='left', on='cases.submitter_id') # adds mutation status to wsi samples

file_names = wsi_filtered['file_name'].tolist() # list of wsi file names to be processed
mutation_status = wsi_filtered['mutation_status'].tolist() # mutation status of wsi samples, order preserved

# Process embeddings ###############################################################################################
mean_embeddings, median_embeddings = processEmbeddings(path_to_embeddings, file_names)
mean_embeddings.to_csv('data/sophie_ML/mean_image_embeddings.csv', index=False)
median_embeddings.to_csv('data/sophie_ML/median_embeddings.csv', index=False)

# Check if mutation_status is correct
# mean_embeddings['mutation'] = mutation_status
# mean_embeddings['cases.submitter_id'] = mean_embeddings.index.str[0:12]
# mut_patients_emb = mean_embeddings[mean_embeddings['mutation'] == 1]['cases.submitter_id'].unique().tolist()
# mut_patients = patients[patients['mutation_status'] == 1]['cases.submitter_id'].unique().tolist()
# sorted(mut_patients_emb) == sorted(mut_patients)

# Dimensionality reduction #########################################################################################
# PCA
pca_mean = dimReduction(mean_embeddings, mutation_status, reduction='PCA')
fig, ax = plotMAP(pca_mean, title='PCA reduction of image embeddings\n(mean values)')
fig.savefig('results/image_embeddings/PCA_mean.png', bbox_inches='tight')
pca_median = dimReduction(median_embeddings, mutation_status, reduction='PCA')
fig, ax = plotMAP(pca_median, title='PCA reduction of image embeddings\n(median values)')
fig.savefig('results/image_embeddings/PCA_median.png', bbox_inches='tight')

# UMAP
umap_mean = dimReduction(mean_embeddings, mutation_status, reduction='UMAP')
fig, ax = plotMAP(umap_mean, title='UMAP reduction of image embeddings\n(mean values)')
fig.savefig('results/image_embeddings/UMAP_mean.png', bbox_inches='tight')
umap_median = dimReduction(median_embeddings, mutation_status, reduction='UMAP')
fig, ax = plotMAP(umap_median, title='UMAP reduction of image embeddings\n(median values)')
fig.savefig('results/image_embeddings/UMAP_median.png', bbox_inches='tight')

# t-SNE
tsne_mean = dimReduction(mean_embeddings, mutation_status, reduction='TSNE')
fig, ax = plotMAP(tsne_mean, title='TSNE reduction of image embeddings\n(mean values)')
fig.savefig('results/image_embeddings/TSNE_mean.png', bbox_inches='tight')
tsne_median = dimReduction(median_embeddings, mutation_status, reduction='TSNE')
fig, ax = plotMAP(tsne_median, title='TSNE reduction of image embeddings\n(median values)')
fig.savefig('results/image_embeddings/TSNE_median.png', bbox_inches='tight')


