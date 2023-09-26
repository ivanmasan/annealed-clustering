from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Callable

import numpy as np

from data_load import load_data, filter_infrequent_skus


@dataclass
class MatrixChange:
    sku_id: int
    cluster_id: int
    delta: float

    def __post_init__(self):
        self.delta = float(self.delta)


@dataclass
class StateChange:
    sku_id: int
    delta: int


@dataclass
class ChangePolicy:
    state: list[int]
    callable: Callable
    weight: float = 1
    extra_kwargs: dict = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

    def suggest(self, cluster_map, sku_idx):
        return self.callable(cluster_map, sku_idx, **self.extra_kwargs)


def simple_change_generator(data, double_sugggestion_ratio):
    cross_idx, cross_probs = generate_cross(data)
    cross_dict = {'cross_idx': cross_idx, 'cross_probs': cross_probs}
    change_policies = [
        ChangePolicy([0], switch_single_cluster, 1 - double_sugggestion_ratio),
        ChangePolicy([0], switch_two_clusters, double_sugggestion_ratio, cross_dict),
    ]
    return ChangeGenerator(change_policies, data.shape[1])


def split_change_generator(data, double_sugggestion_weight, split_weight, merge_prob):
    cross_idx, cross_probs = generate_cross(data)
    cross_dict = {'cross_idx': cross_idx, 'cross_probs': cross_probs}
    change_policies = [
        ChangePolicy([0], switch_single_cluster, 1),
        ChangePolicy([0], switch_two_clusters, double_sugggestion_weight, cross_dict),
        ChangePolicy([0], split_sku, split_weight),
        ChangePolicy([1], merge_sku, merge_prob),
        ChangePolicy([1], move_split_sku, 1 - merge_prob)
    ]
    return ChangeGenerator(change_policies, data.shape[1])


class ChangeGenerator:
    def __init__(self, change_policies, sku_count):
        self.sku_count = sku_count

        self.policies_dict = defaultdict(list)
        self.policies_probs = defaultdict(list)
        for policy in change_policies:
            for state in policy.state:
                self.policies_dict[state].append(policy)
                self.policies_probs[state].append(policy.weight)

        for k, v in self.policies_probs.items():
            self.policies_probs[k] = np.array(v) / sum(v)

        self.sku_idx = []

    def _fill_skus_to_switch(self):
        self.sku_idx = list(range(self.sku_count))
        np.random.shuffle(self.sku_idx)

    def suggest_changes(self, cluster_map, cluster_state):
        if not self.sku_idx:
            self._fill_skus_to_switch()

        sku_idx = self.sku_idx.pop()
        state = cluster_state[sku_idx]

        policy = np.random.choice(self.policies_dict[state], p=self.policies_probs[state])

        return policy.suggest(cluster_map, sku_idx)


def switch_single_cluster(cluster_map, sku_idx):
    clusters = cluster_map.shape[0]
    orig_cluster = np.where(cluster_map[:, sku_idx] == 1)[0][0]
    new_cluster = (orig_cluster + 1 + np.random.randint(clusters - 1)) % clusters

    return [
        MatrixChange(sku_idx, orig_cluster, -1),
        MatrixChange(sku_idx, new_cluster, 1),
    ], []


def add_additional_pair_change(cluster_map, changes, cross_idx, cross_probs):
    sku_idx = changes[0].sku_id
    orig_cluster = changes[0].cluster_id
    new_cluster = changes[1].cluster_id

    mask = (
        (cluster_map[orig_cluster, cross_idx[sku_idx]] > 0)
        & (cluster_map[new_cluster, cross_idx[sku_idx]] == 0)
    )

    if sum(mask) == 0:
        return []

    probs = cross_probs[sku_idx][mask]
    probs_sum = probs.sum()
    probs = probs / probs_sum

    second_sku_idx = np.random.choice(cross_idx[sku_idx][mask], p=probs)

    value = cluster_map[orig_cluster, second_sku_idx]
    return [
        MatrixChange(second_sku_idx, orig_cluster, -value),
        MatrixChange(second_sku_idx, new_cluster, value)
    ]


def switch_two_clusters(cluster_map, sku_idx, cross_idx, cross_probs):
    changes, _ = switch_single_cluster(cluster_map, sku_idx)
    additional_changes = add_additional_pair_change(
        cluster_map, changes, cross_idx, cross_probs)
    return changes + additional_changes, []


def split_sku(cluster_map, sku_idx):
    clusters = cluster_map.shape[0]
    orig_cluster = np.where(cluster_map[:, sku_idx] == 1)[0][0]
    new_cluster = (orig_cluster + 1 + np.random.randint(clusters - 1)) % clusters

    return [
        MatrixChange(sku_idx, orig_cluster, -0.5),
        MatrixChange(sku_idx, new_cluster, 0.5),
    ], [StateChange(sku_idx, 1)]


def merge_sku(cluster_map, sku_idx):
    clusters = np.where(cluster_map[:, sku_idx] > 0)[0]
    np.random.shuffle(clusters)

    return [
        MatrixChange(sku_idx, clusters[0], -0.5),
        MatrixChange(sku_idx, clusters[1], 0.5)
    ], [StateChange(sku_idx, -1)]


def move_split_sku(cluster_map, sku_idx):
    clusters = np.where(cluster_map[:, sku_idx] > 0)[0]
    new_clusters = np.arange(cluster_map.shape[0])
    new_clusters = np.delete(new_clusters, clusters)

    return [
        MatrixChange(sku_idx, np.random.choice(clusters), -0.5),
        MatrixChange(sku_idx, np.random.choice(new_clusters), 0.5)
    ], []


def generate_cross(data, top_n_similiar=20, prob_smoothness=0.02):
    cross = (data.T @ data).A
    cross = cross / np.diag(cross).reshape(-1, 1)
    cross_idx = np.argsort(cross, axis=1)[:, -top_n_similiar:-1]
    cross_val = np.take_along_axis(cross, cross_idx, axis=1)

    cross_probs = np.exp(cross_val / prob_smoothness)

    return cross_idx, cross_probs




if False:
    cluster_map = np.full(shape=(4, 8), fill_value=0.0)
    for i in range(8):
        cluster_map[np.random.randint(cluster_map.shape[0]), i] = 1.0

    cluster_state = np.full(8, 1)

    for _ in range(3):
        idx = np.random.choice(np.arange(8))
        cluster_state[idx] = 2
        cluster_map[:, idx] = 0

        i, j = np.random.choice(np.arange(cluster_map.shape[0]), 2, replace=False)

        cluster_map[i, idx] = 0.5
        cluster_map[j, idx] = 0.5

    cluster_map.sum(0)


    data, sku_vals = load_data()
    data, _ = filter_infrequent_skus(data, 1600)

    generator = ChangeGenerator(data)

    for i in range(1000):
        matrix_changes, state_changes = generator.suggest_changes(cluster_map, cluster_state)
        print(cluster_map)
        for change in matrix_changes:
            cluster_map[change.cluster_id, change.sku_id] += change.delta
        for change in state_changes:
            cluster_state[change.sku_id] = change.state

        assert np.all(cluster_map.sum(0) == 1)
