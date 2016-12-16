import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np

from embedded_jubatus import Clustering
from jubatus.clustering.types import WeightedDatum
from jubatus.clustering.types import WeightedIndex
from jubatus.clustering.types import IndexedPoint
from jubatus.common import Datum

n_samples = 300
np.random.seed(0)

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8, centers=2)
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

datasets_list = [
    StandardScaler().fit_transform(X[0])
    for X in [noisy_circles, noisy_moons, blobs]
]
datasets_name = ["noisy_circles", "noisy_moons", "blobs"]

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

CONFIG = {
    'method': 'kmeans',
    'parameter': {
        'k' : 2,
        'seed' : 0,
    },
    'compressor_parameter': {
        'bucket_size': 2,
    },
    'compressor_method' : 'simple',
    'converter': {
        'num_filter_types': {},
        'num_filter_rules': [],
        'string_filter_types': {},
        'string_filter_rules': [],
        'num_types': {},
        'num_rules': [
            {'key': '*', 'type': 'num'}
        ],
        'string_types': {},
        'string_rules': [
            {'key': '*', 'type': 'space',
             'sample_weight': 'bin', 'global_weight': 'bin'}
        ]
    },
}
clients = [Clustering(CONFIG) for _ in datasets_list]

fig, axs = plt.subplots(1, len(datasets_list), figsize=(len(datasets_list) * 2 + 6, 4))

def draw_decision_surface(i):
    print('iteration:', i)
    for client, ax, X in zip(clients, axs, datasets_list):
        if i == 0:
            client.clear()
        client.push([IndexedPoint(str(i), Datum({'x' : X[i, 0], 'y' : X[i, 1]}))])
        if i <= 2:
            continue
        centers = client.get_k_center()
        clusters = client.get_core_members_light()
        ax.clear()
        for y, cluster in enumerate(clusters):
            for weighted_index in cluster:
                ax.scatter(X[int(weighted_index.id), 0], X[int(weighted_index.id), 1], color=colors[y].tolist(), s=10)
        for y, center in enumerate(centers):
            ax.scatter(center.num_values[0][1], center.num_values[1][1], color=colors[y].tolist(), s=50)

ani = FuncAnimation(fig, draw_decision_surface, frames=n_samples, interval=50, repeat=False)
ani.save('test.gif', writer='imagemagick')
plt.show()
