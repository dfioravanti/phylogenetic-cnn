import os
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

taxonomy_ranks = OrderedDict()
taxonomy_ranks['k__'] = 'kingdom'
taxonomy_ranks['p__'] = 'phylum'
taxonomy_ranks['c__'] = 'class'
taxonomy_ranks['o__'] = 'order'
taxonomy_ranks['f__'] = 'family'
taxonomy_ranks['g__'] = 'genus'


def empty_taxonomy():
    taxonomy = OrderedDict()

    taxonomy['kingdom'] = None
    taxonomy['phylum'] = None
    taxonomy['class'] = None
    taxonomy['order'] = None
    taxonomy['family'] = None
    taxonomy['genus'] = None

    return taxonomy


def split_taxonomy(bacteria):
    splited_taxonomy = bacteria.split('.')
    fixed_taxonomy = []

    for element in splited_taxonomy:
        if element.startswith('__', 1) or element == 'Other':
            fixed_taxonomy.append(element)
        else:
            fixed_taxonomy[-1] = '.'.join([fixed_taxonomy[-1], element])

    return fixed_taxonomy


def extract_taxonomy(bacteria):

    list_taxonomy_ranks = split_taxonomy(bacteria)
    taxonomy = empty_taxonomy()
    for rank in list_taxonomy_ranks:
        if not rank == 'Other':
            correnct_rank = taxonomy_ranks[rank[0:3]]
            if not rank[3:] == '':
                taxonomy[correnct_rank] = rank[3:]

    return taxonomy

DISEASE = 'CDf'
DATA_DIR = 'true_data'

# ============================================
# -- DATA loading section
# ============================================

DISEASE_FOLDER = ''.join(['HS_', DISEASE])

TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, DISEASE_FOLDER,
                                      ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training.txt']))
COORDINATES_FILEPATH = os.path.join(DATA_DIR, 'coordinates',
                                    ''.join(['coordinates_', DISEASE.lower(), '.txt']))
TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, DISEASE_FOLDER,
                                        ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training_lab.txt']))
TEST_DATA_FILEPATH = os.path.join(DATA_DIR, DISEASE_FOLDER,
                                  ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test.txt']))
TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, DISEASE_FOLDER,
                                    ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test_lab.txt']))

training = pd.read_csv(TRAINING_DATA_FILEPATH, sep='\t', index_col=0,)
training_lab = pd.read_csv(TRAINING_LABELS_FILEPATH, sep='\t',dtype=np.int, names=' ')
test = pd.read_csv(TEST_DATA_FILEPATH, sep='\t', index_col=0)
test_lab = pd.read_csv(TEST_LABELS_FILEPATH, sep='\t',dtype=np.int, names=' ')

# ============================================
# -- Process samples to extract taxonomy
# ============================================

# G = nx.DiGraph()
G = nx.Graph()

bacterias = training.columns

for bacteria in bacterias:
    taxonomy = extract_taxonomy(bacteria)
    previous_name = None
    for n, (rank, name) in enumerate(taxonomy.items()):
        if name is not None:
            G.add_node(name)
            if not n == 0 and previous_name is not None:
                G.add_edge(previous_name, name)

        previous_name = name

a = nx.adjacency_matrix(G)
print(a)
plt.spy(a, precision=0.01, markersize=1)
plt.show()
# pos=nx.nx.nx_pydot.graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=True, arrows=False)
figure = plt.gcf()
figure.set_size_inches(4, 3)
# plt.show()
plt.savefig('2.jpg')

