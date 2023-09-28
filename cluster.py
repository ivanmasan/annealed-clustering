from pathlib import Path

import numpy as np

from annealing import Annealing, loss
from data_load import load_data, filter_infrequent_skus, sample_data, sku_id2name
from evaluator import Evaluator

clusters = 8
min_sku_freq = 20
valid_sample_count = 1000000
initial_temperature = 1500000
temperature_decay = 0.999998
annealing_steps = 20000


data, sku_vals = load_data()
filtered_data, sku_mask = filter_infrequent_skus(data, min_sku_freq)
filtered_sku_vals = sku_vals[sku_mask]


annealing = Annealing(
    clusters=clusters,
    T=initial_temperature,
    decay=temperature_decay,
    valid_sample_count=valid_sample_count,
    data=filtered_data,
    use_split_clusters=True,
    split_weight=0.3,
    double_change_weight=0.4,
    merge_prob=0.2,
    cluster_state_reg=0.4,
    split_value=0.4,
    bin_delta=0.2
)

annealing.anneal(annealing_steps, verbose=True)

#evaluator = Evaluator(cluster_map, filtered_sku_vals, sku_id2name(),
#                      Path('images'), filtered_data)
