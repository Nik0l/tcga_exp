import umap
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_luad_data_info(path):
    """ Get information on the image embeddings.
    There are two files, one with the actual embeddings ('embeddings.npy'),
                         the second is with the slide and the patch position information ('paths.pkl')"""

    data = np.load(path + 'embeddings.npy')
    print('Embeddings shape:', data.shape)

    data_info = pd.read_pickle(path + 'paths.pkl')

    image_names = [f.split('/')[2] for f in data_info if f.split('/')[1] == 'TCGA-LUAD']
    #print(image_names)
    image_names_unique = list(set(image_names))
    print(image_names_unique)

    n_patches_total = len(image_names)

    unique_names = set(image_names)
    n_images_total = len(unique_names)

    print('Total patches:', n_patches_total)
    print('Total unique images:', n_images_total)


def get_patches_for_image(path, slide_name):
    """ Get all patches for a slide of interest."""
    print(slide_name)
    data_info = pd.read_pickle(path + 'paths.pkl')
    print(type(data_info))
    filter_indices = [i for i, s in enumerate(data_info) if slide_name in s]
    n_patches_total = len(filter_indices)
    print('Total %d patches in %s image' % (n_patches_total, slide_name))
    data = np.load(path + 'embeddings.npy')
    print(data.shape)
    patches = np.take(data, filter_indices, axis=0)
    return patches


def plot_umap_samples(samples):
    """ Reduce dimension to two using UMAP and plot the samples, for example, embeddings."""
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(samples)
    print(embedding.shape)
    df_embedding = pd.DataFrame({'x1': embedding[:, 0], 'x2': embedding[:, 1]})

    # di = {'1.0': 'red', '0.0': 'blue'}
    # df_embedding['colour'] = df_embedding[column_name].map(di)
    # plt.scatter(df_embedding['x1'], df_embedding['x2'], s=5, c=df_embedding['colour'], cmap=plt.cm.autumn)
    plt.scatter(df_embedding['x1'], df_embedding['x2'], s=5, cmap=plt.cm.autumn)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the image embeddings', fontsize=18)

    #plt.savefig(fig_file)
    plt.show()


def process_second_embedding_data(path='/Users/kmwx127/PycharmProjects/project/tcga_exp/data/image_embeddings/'):

    get_luad_data_info(path)
    patches = get_patches_for_image(path, slide_name='5eeb2ac5e4b0b6e434491680.svs')
    plot_umap_samples(patches)


def get_embeddings_image(path, image_name):
    """ Load all the embedded patches from the slide. """
    image_embeddings = np.loadtxt(path + image_name + '.csv', delimiter=',', skiprows=1)
    n_patches = image_embeddings.shape[0]
    print('Total %d patches in %s slide' % (n_patches, image_name))
    return image_embeddings


def process_first_embedding_data(path):
    luad_images_only = get_first_luad_data_info(path)
    image_embeddings = get_embeddings_image(path, luad_images_only[1])
    print(image_embeddings)
    plot_umap_samples(image_embeddings)


def get_first_luad_data_info(path):
    """ Get only LUAD images names. """
    luad_images_only = pd.read_csv(path.rsplit('/', 2)[0] + '/LUAD.csv')['0']
    luad_images_only = list(set([f.split('/')[1] for f in list(luad_images_only)]))
    print('total LUAD images in the list:', len(luad_images_only))
    return luad_images_only


path_one = '/Users/kmwx127/Downloads/public_method/tcga-dataset/tcga_lung_data_feats/'
#process_first_embedding_data(path=path_one)

path_two = '/Users/kmwx127/PycharmProjects/project/tcga_exp/data/image_embeddings/'
#process_second_embedding_data(path=path_two)
get_luad_data_info(path=path_two)
path_tree = '~/PycharmProjects/project/tcga_exp/data/'

df_mapping = pd.read_csv(path_tree + 'tcga_mapping.txt', delimiter='\t')
#print(df_mapping)
