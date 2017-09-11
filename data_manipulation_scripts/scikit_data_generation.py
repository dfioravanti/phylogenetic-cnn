import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

DISEASE = 'CDf'
total_nb_samples = 1000
DATA_DIR = '../datasets'
plot = False

OUTPUT_PATH = os.path.join(DATA_DIR, 'scikit_data', DISEASE, str(total_nb_samples))
DISEASE_FOLDER = ''.join(['HS_', DISEASE])
TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, 'true_data', DISEASE_FOLDER,
                                      ''.join(['Sokol_16S_taxa_HS_', DISEASE,
                                               '_commsamp_training.txt']))

true_sample = pd.read_csv(TRAINING_DATA_FILEPATH, sep='\t', header=0, index_col=0)
bacterias_name = true_sample.columns.values

X, y = make_classification(n_samples=total_nb_samples, n_features=259, n_informative=237, n_redundant=20,
                           random_state=1, n_clusters_per_class=1,  class_sep=1, weights=[0.37, 0.63])

healty = X[y == 0]
sick = X[y == 1]

if plot:
    pca = PCA(n_components=2)
    pca.fit(X)
    new_X = pca.transform(X)
    fig = plt.figure("Test", facecolor='none')
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.clf()
    healty_plt = plt.scatter(healty[:, 0], healty[:, 1], c='r', cmap=cm_bright, alpha=0.3)
    sick_plt = plt.scatter(sick[:, 0], sick[:, 1], c='b', cmap=cm_bright, alpha=0.3)
    legend = plt.legend((healty_plt, sick_plt), ('Healthy', 'Sick'))
    frame = legend.get_frame()
    frame.set_linewidth(0)
    frame.set_facecolor('none')
    fig.patch.set_visible(False)
    plt.axis('on')
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = pd.DataFrame(X_train, columns=bacterias_name)
X_test = pd.DataFrame(X_test, columns=bacterias_name)


os.makedirs(OUTPUT_PATH, exist_ok=True)
X_train.to_csv(os.path.join(OUTPUT_PATH, 'training.txt'), sep='\t')
y_train.tofile(os.path.join(OUTPUT_PATH, 'training_labels.txt'), sep='\n')
X_test.to_csv(os.path.join(OUTPUT_PATH, 'test.txt'), sep='\t')
y_test.tofile(os.path.join(OUTPUT_PATH, 'test_labels.txt'), sep='\n')
