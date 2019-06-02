import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import utils


def get_argparser():
    parser = argparse.ArgumentParser(description=os.path.basename(__file__))
    parser.add_argument('-input', default='./data', help='input data dir path')
    parser.add_argument('-k', type=int, required=False, help='[optional] fixed cluster size')
    parser.add_argument('-features', default='sparse', help='features for clustering: sparse, pretrained, or custom')
    parser.add_argument('-output', default='./', help='output dir path')
    return parser.parse_args()


def split_feature_mat(feature_mat):
    sample_size = feature_mat.shape[0]
    return feature_mat[:int(sample_size / 2)], feature_mat[int(sample_size / 2):]


def clustering(train_feature_mat, val_feature_mat, k):
    kmeans = MiniBatchKMeans(n_clusters=k)
    kmeans.fit(train_feature_mat)
    clusters = kmeans.predict(val_feature_mat)
    return clusters, kmeans.inertia_


def plot_k_vs_inertia(feature_mat, output_file_path=None):
    k_list = list(range(2, 30))
    inertia_list = list()
    train_feature_mat, val_feature_mat = split_feature_mat(feature_mat)
    for k in k_list:
        clusters, inertia = clustering(train_feature_mat, val_feature_mat, k)
        inertia_list.append(inertia)
    plt.plot(k_list, inertia_list)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.xticks(np.arange(min(k_list), max(k_list) + 1, 2.0))
    if output_file_path is not None:
        plt.savefig(output_file_path)
    else:
        plt.show()


def write_clusters(indices, clusters, dir_path, file_name='clusters.csv'):
    with open(os.path.join(dir_path, file_name), 'w') as fp:
        for i in range(len(indices)):
            fp.write(indices[i] + ',' + str(clusters[i]) + '\n')


def purity_score(c, y):
    A = np.c_[(c,y)]
    n_accurate = 0.
    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
    return n_accurate / A.shape[0]


if __name__ == "__main__":
    args = get_argparser()
    input_dir_path = args.input
    output_dir_path = args.output
    print("Reading data")
    indices, reviews, labels = utils.get_training_data(input_dir_path)
    print("Training model")
    feature_mat_file_path = 'reviews_features_sparse.pkl' if args.features == 'sparse'\
        else 'reviews_features_dense_pretrained.pkl' if args.features == 'pretrained'\
        else 'reviews_features_dense_custom.pkl'
    feature_mat = pickle.load(open(feature_mat_file_path, 'rb'))
    if args.k is None:
        plot_k_vs_inertia(feature_mat, args.features + '_k_vs_inertia.png')
    else:
        train_feature_mat, val_feature_mat = split_feature_mat(feature_mat)
        train_size = train_feature_mat.shape[0]
        clusters, inertia = clustering(train_feature_mat, val_feature_mat, args.k)
        print("Writing clusters")
        write_clusters(indices[train_size:], clusters, output_dir_path, file_name=args.features + '_clusters.csv')
        purity = purity_score(clusters, labels[train_size:])
        print(purity)
