import itertools
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.sparse import coo_matrix


class ClusterMap:
    def __init__(self, data, clusters, split_proportions=0):
        self._set_bins()
        self.data = data

        self._init_cluster_map(clusters, data.shape[1], split_proportions)
        self.cluster_data = data @ self.cluster_map.T

        self.clusters = self.cluster_map.shape[0]
        self.sku_count = self.cluster_map.shape[1]

        self.digitized_cluster_data = np.digitize(self.cluster_data, self.bins)
        self.digitized_cluster_data_counts = np.zeros((len(self.bins) + 1, self.clusters))

        for i in range(self.clusters):
            items, counts = np.unique(self.digitized_cluster_data[:, i], return_counts=True)
            self.digitized_cluster_data_counts[items, i] = counts

        self.cluster_counts = self.cluster_data.sum(0)

    def _set_bins(self):
#        self.bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
#        self.bin_delta = 1
#        self.bin_diffs = np.linspace(-6, 6, 13)
        self.bins = np.linspace(0.25, 5.75, 12)
        self.bin_delta = 0.5
        self.bin_diffs = np.linspace(-6, 6, 25)

    def _init_cluster_map(self, clusters, skus, split_proportions):
        self.cluster_map = np.full(shape=(clusters, skus), fill_value=0.0)
        self.state_map = np.full(skus, 0)

        for i in range(skus):
            if np.random.rand() < split_proportions:
                k, l = np.random.choice(np.arange(clusters), 2, replace=False)
                self.cluster_map[k, i] = 0.5
                self.cluster_map[l, i] = 0.5
                self.state_map[i] = 1
            else:
                self.cluster_map[np.random.randint(clusters), i] = 1

    def apply_changes_calculate_loss(self, changes):
        loss = 0
        for change in changes:
            loss += self.calculate_loss_simple(change)
            self.apply_change(change)
        return loss

    def apply_change(self, change):
        self.cluster_map[change.cluster_id, change.sku_id] += change.delta
        assert 1 >= self.cluster_map[change.cluster_id, change.sku_id] >= 0

        delta_cluster_map = coo_matrix(
            ([change.delta], ([change.sku_id], [change.cluster_id])),
            shape=(self.sku_count, self.clusters))
        delta_cluster_map = delta_cluster_map.tocsr()

        data_delta = self.data @ delta_cluster_map
        self.cluster_counts += data_delta.sum(0).A.flatten()
        self.cluster_data += data_delta

        data_delta = data_delta.tocoo()
        rows = data_delta.row
        cols = data_delta.col

        bins = np.digitize(self.cluster_data[rows, cols], self.bins)
        old_vals = self.digitized_cluster_data[rows, cols]
        self.digitized_cluster_data[rows, cols] = bins

        old_items, old_counts = np.unique(old_vals, return_counts=True)
        new_items, new_counts = np.unique(bins, return_counts=True)

        self.digitized_cluster_data_counts[old_items, cols[0]] -= old_counts
        self.digitized_cluster_data_counts[new_items, cols[0]] += new_counts

        self.cluster_counts[change.cluster_id] += change.delta

    def apply_state_change(self, state_change):
        self.state_map[state_change.sku_id] += state_change.delta

    def calculate_loss(self, changes):
        changes_by_cluster = defaultdict(list)

        for change in changes:
            changes_by_cluster[change.cluster_id].append(change)

        total_loss_change = 0

        for cluster_changes in changes_by_cluster.values():
            cluster_loss = self.calculate_cluster_loss(cluster_changes)
            total_loss_change += cluster_loss

        return total_loss_change

    def calculate_cluster_loss(self, changes):
        indices = []
        for change in changes:
            sku_mask = self.data[:, change.sku_id]
            indices.append(set(sku_mask.tocoo().row))

        all_counts = self.digitized_cluster_data_counts[:, changes[0].cluster_id]

        counts_dict = {}
        for values in itertools.product(*[[0, 1]] * len(changes)):
            if np.all(np.array(values) == 0):
                continue

            subset_indices = set()
            for i, v in enumerate(values):
                if v == 1:
                    if subset_indices:
                        subset_indices &= indices[i]
                    else:
                        subset_indices = deepcopy(indices[i])

            for i, v in enumerate(values):
                if v == 0:
                    subset_indices -= indices[i]

            bin_items = self.digitized_cluster_data[list(subset_indices), changes[0].cluster_id]

            items, counts = np.unique(bin_items, return_counts=True)
            bin_counts = np.zeros(all_counts.shape)
            bin_counts[items] = counts

            counts_dict[values] = bin_counts

        counts_dict[tuple([0] * len(changes))] = all_counts - sum(counts_dict.values())

        total_change = 0
        change_vector = np.array([c.delta for c in changes])

        for left_values in itertools.product(*[[0, 1]] * len(changes)):
            left_values_array = np.array(left_values)
            for right_values in itertools.product(*[[0, 1]] * len(changes)):
                right_values_array = np.array(right_values)

                if np.all(left_values_array == 0) and np.all(right_values_array == 0):
                    continue

                diff_counts = np.convolve(counts_dict[left_values], counts_dict[right_values][::-1])

                left_change = (change_vector * left_values_array).sum()
                right_change = (change_vector * right_values_array).sum()

                loss_delta = (
                    np.minimum(self.bin_diffs + left_change, right_change)
                    - np.minimum(self.bin_diffs, 0)
                )

                total_change += (loss_delta * diff_counts).sum()
        return total_change

    def calculate_loss_simple(self, change):
        sku_mask = self.data[:, change.sku_id]
        indices = sku_mask.tocoo().row

        positive_bins = self.digitized_cluster_data[indices, change.cluster_id]
        all_counts = self.digitized_cluster_data_counts[:, change.cluster_id]

        items, counts = np.unique(positive_bins, return_counts=True)
        positive_counts = np.zeros(all_counts.shape)
        positive_counts[items] = counts

        p, a = np.meshgrid(positive_counts, all_counts - positive_counts)
        combinations = p * a

        larger = np.tril(combinations, -1).sum()
        equal = np.tril(combinations).sum() - larger

        return (
            larger * 2
            + equal * (change.delta < 0) * 2
            + sku_mask.nnz ** 2
        ) * change.delta

    def calculate_exact_loss(self, change):
        sku_mask = self.data[:, change.sku_id]
        indices = sku_mask.tocoo().row
        inv = [x for x in np.arange(len(self.cluster_data))
               if x not in indices]

        positive_counts = self.cluster_data[indices, change.cluster_id]
        negative_counts = self.cluster_data[inv, change.cluster_id]

        positive_counts = np.sort(positive_counts)
        negative_counts = np.sort(negative_counts)

        sort_idx = np.searchsorted(negative_counts, positive_counts, side='left')
        smaller = sort_idx.sum()

        sort_idx = np.searchsorted(negative_counts, positive_counts, side='right')
        equal_or_smaller = sort_idx.sum()

        larger = len(positive_counts) * len(negative_counts) - equal_or_smaller
        equal = equal_or_smaller - smaller

        return (
            larger * 2
            + equal * (change.delta < 0) * 2
            + sku_mask.nnz ** 2
        ) * change.delta
