from pathlib import Path

from annealing import Annealing
from data_load import load_data, filter_infrequent_skus, sample_data, sku_id2name
from evaluator import Evaluator

clusters = 8
min_sku_freq = 20
valid_sample_count = 100000
initial_temperature = 1000000
temperature_decay = 0.9999
annealing_steps = 10000


data, sku_vals = load_data()
filtered_data, sku_mask = filter_infrequent_skus(data, min_sku_freq)
filtered_sku_vals = sku_vals[sku_mask]


annealing = Annealing(
    clusters=clusters,
    T=initial_temperature,
    decay=temperature_decay,
    valid_sample_count=valid_sample_count,
    data=filtered_data,
    initial_split_proportion=0,
    use_split_clusters=True,
    split_weight=0.4,
    double_change_weight=0.4,
    merge_prob=0.4,
    cluster_state_reg=0.5
)

annealing.anneal(annealing_steps, verbose=True)
cluster_map = annealing.get_cluster_map()
cluster_state = annealing.cluster_map.state_map
cluster_state.mean()

#evaluator = Evaluator(cluster_map, filtered_sku_vals, sku_id2name(),
#                      Path('images'), filtered_data)
