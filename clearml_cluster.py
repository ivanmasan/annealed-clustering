from pathlib import Path

import numpy as np

from annealing import Annealing
from cross_matrix import generate_convolution_pattern, generate_cross_matrix
from data_load import load_data, filter_infrequent_skus, sample_data, sku_id2name
from evaluator import Evaluator

from clearml import Task, Logger


task = Task.current_task()
if task is None:
    task = Task.init(project_name='clustering/anneal', task_name="Test Task 4")

logger = Logger.current_logger()


params = {
    'clusters': 13 * 2,
    'grid_shape': (13, 2),
    'convolutions': {
        'center': 3,
        'neighbour': 1,
        'diag': 0,
        'neighbour_2': 0
    },
    'min_sku_freq': 20,
    'valid_sample_count': 1000000,
    'initial_temperature': 1500000,
    'temperature_decay': 0.999998,
    'annealing_steps': 2000000,
    'double_change_chance': 0.4,
    'use_split_clusters': False,
    'split_weight': 0.3,
    'merge_prob': 0.2,
    'cluster_state_reg': 0.0,
    'split_value': 0,
    'bin_delta': 1,
    'max_steps': 15
}
params = task.connect(params)


data, sku_vals = load_data()
filtered_data, sku_mask = filter_infrequent_skus(data, params['min_sku_freq'])
filtered_sku_vals = sku_vals[sku_mask]

if params['grid_shape']:
    conv_pattern = generate_convolution_pattern(**params['convolutions'])
    cross_matrix = generate_cross_matrix(params['grid_shape'], conv_pattern)
    assert params['clusters'] == np.prod(params['grid_shape'])
else:
    cross_matrix = None

annealing = Annealing(
    clusters=params['clusters'],
    T=params['initial_temperature'],
    decay=params['temperature_decay'],
    valid_sample_count=params['valid_sample_count'],
    data=filtered_data,
    use_split_clusters=params['use_split_clusters'],
    split_weight=params['split_weight'],
    double_change_weight=params['double_change_chance'],
    merge_prob=params['merge_prob'],
    cluster_state_reg=params['cluster_state_reg'],
    bin_delta=params['bin_delta'],
    split_value=params['split_value'],
    max_steps=params['max_steps'],
    cross_cluster_map=cross_matrix
)

annealing.anneal(params['annealing_steps'], logger=logger)
cluster_map = annealing.get_cluster_map()

evaluator = Evaluator(cluster_map, filtered_sku_vals, sku_id2name(),
                      Path('images'), filtered_data)

evaluator.clearml_image_summary(logger)
logger.report_table(
    "Summary Table",
    "-",
    table_plot=evaluator.data_summary()
)

task.upload_artifact("Cluster Map", artifact_object=cluster_map)
task.upload_artifact("Sku Vector", artifact_object=filtered_sku_vals)

task.close()
