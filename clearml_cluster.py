from pathlib import Path

from annealing import Annealing
from data_load import load_data, filter_infrequent_skus, sample_data, sku_id2name
from evaluator import Evaluator

from clearml import Task, Logger


task = Task.current_task()
if task is None:
    task = Task.init(project_name='clustering/anneal', task_name="Test Task")

logger = Logger.current_logger()


params = task.connect({
    'clusters': 8,
    'min_sku_freq': 20,
    'valid_sample_count': 100000,
    'initial_temperature': 1500000,
    'temperature_decay': 0.999998,
    'annealing_steps': 2000000,
    'double_change_chance': 0.4,
    'use_split_clusters': True,
    'split_weight': 0.3,
    'merge_prob': 0.2,
    'cluster_state_reg': 0,
    'split_value': 0.4,
    'bin_delta': 0.2
})


data, sku_vals = load_data()
filtered_data, sku_mask = filter_infrequent_skus(data, params['min_sku_freq'])
filtered_sku_vals = sku_vals[sku_mask]


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
    cluster_state_reg=params['cluster_state_reg']
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
