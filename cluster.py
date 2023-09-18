from pathlib import Path

from annealing import Annealing
from data_load import load_data, filter_infrequent_skus, sample_data, sku_id2name
from evaluator import Evaluator

clusters = 8
min_sku_freq = 0
valid_sample_count = 100000
initial_temperature = 1000000
temperature_decay = 0.999997
annealing_steps = 1000000


data, sku_vals = load_data()
filtered_data, sku_mask = filter_infrequent_skus(data, min_sku_freq)
filtered_sku_vals = sku_vals[sku_mask]


annealing = Annealing(
    clusters=clusters,
    T=initial_temperature,
    decay=temperature_decay,
    valid_sample_count=valid_sample_count,
    data=data
)

annealing.anneal(annealing_steps)
cluster_map = annealing.get_cluster_map()

evaluator = Evaluator(cluster_map, sku_vals, sku_id2name(), Path('images'), data)
