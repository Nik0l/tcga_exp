import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_luad_data_info(path):
    """ Get information on the image embeddings.
    There are two files, one with the actual embeddings ('embeddings.npy'),
                         the second is with the slide and the patch position information ('paths.pkl')"""

    data = np.load(path + 'embeddings.npy')
    print(data.shape)

    data_info = pd.read_pickle(path + 'paths.pkl')
    print(data_info[0])

    image_names = [f.split('/')[2] for f in data_info if f.split('/')[1] == 'TCGA-LUAD']

    n_patches_total = len(image_names)

    unique_names = set(image_names)
    n_images_total = len(unique_names)

    print(n_patches_total)
    print(n_images_total)


def get_patches_for_image(path, slide_name):
    """ Get all patches for a slide of interest."""
    print(slide_name)
    data_info = pd.read_pickle(path + 'paths.pkl')
    print(type(data_info))
    filter_indices = [i for i, s in enumerate(data_info) if slide_name in s]
    #print(filter_indices)
    n_patches_total = len(filter_indices)
    print('total patches in slide', n_patches_total)
    data = np.load(path + 'embeddings.npy')
    print(data.shape)
    patches = np.take(data, filter_indices, axis=0)
    print(patches.shape)
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


path = '/../tcga_exp/data/image_embeddings/'
get_luad_data_info(path)
patches = get_patches_for_image(path, image_name='5eeb2ac5e4b0b6e434491680.svs')
plot_umap_samples(patches)

