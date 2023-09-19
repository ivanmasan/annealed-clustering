import numpy as np

from changes import MatrixChange
from cluster_map import ClusterMap
from data_load import sample_data


class Annealing:
    def __init__(self, clusters, T, decay, valid_sample_count, data, double_change_chance=0):
        self.cluster_map = ClusterMap(data, clusters)
        self.double_change_chance = double_change_chance
        self.T = T
        self.decay = decay
        self.valid_orders = (
            sample_data(data, valid_sample_count),
            sample_data(data, valid_sample_count)
        )

        cross = (data.T @ data).A
        cross = cross / np.diag(cross).reshape(-1, 1)
        self.cross_idx = np.argsort(cross, axis=1)[:, -20:-1]
        cross_val = np.take_along_axis(cross, self.cross_idx, axis=1)
        cross_probs = np.exp(cross_val / 0.02)
        self.cross_probs = cross_probs / cross_probs.sum(1, keepdims=True)

        self.improvement_rate = 0
        self.acceptance_rate = 0
        self.loss_delta = 0
        self.iterations = 0

    def anneal_once(self):
        self.acceptance_rate *= 0.995
        self.improvement_rate *= 0.995
        self.loss_delta *= 0.995
        self.iterations += 1

        changes = self.generate_changes()
        exp_change = self.cluster_map.calculate_loss(changes)

        if exp_change < 0 or np.exp(-exp_change / self.T) > np.random.rand():
            self.cluster_map.apply_change(changes[0])
            self.cluster_map.apply_change(changes[1])
            self.acceptance_rate += 0.005
            self.improvement_rate += 0.005 * (exp_change < 0)
            self.loss_delta += 0.005 * exp_change

        self.T *= self.decay

    def generate_changes(self):
        cluster_map = self.cluster_map.cluster_map
        clusters = cluster_map.shape[0]
        sku_idx = np.random.randint(cluster_map.shape[1])
        original_cluster_id = np.where(cluster_map[:, sku_idx])[0][0]
        new_cluster_id = (original_cluster_id + 1 + np.random.randint(clusters - 1)) % clusters

        changes = [
            MatrixChange(sku_idx, original_cluster_id, -1),
            MatrixChange(sku_idx, new_cluster_id, 1),
        ]

        if np.random.rand() < self.double_change_chance:
            mask = cluster_map[original_cluster_id, self.cross_idx[sku_idx]]
            if sum(mask) > 0:
                probs = self.cross_probs[sku_idx][mask]
                probs = probs / probs.sum()
                second_sku_idx = np.random.choice(self.cross_idx[sku_idx][mask], p=probs)
                changes.extend([
                    MatrixChange(second_sku_idx, original_cluster_id, -1),
                    MatrixChange(second_sku_idx, new_cluster_id, 1)
                ])

        return changes

    def anneal(self, iters, verbose=0, logger=None):
        for _ in range(iters):
            self.anneal_once()
            if self.iterations % 1000 == 0:
                if verbose:
                    self._print_status()
                if logger is not None:
                    self._report_status(logger)

    def _generate_metrics(self):
        return {
            "Iterations": self.iterations,
            "Loss": loss(self.valid_orders[0], self.valid_orders[1], self.cluster_map.cluster_map)
                     / self.valid_orders[0].shape[0],
            "Improvement Rate": self.improvement_rate,
            "Acceptance Rate": (self.acceptance_rate - self.improvement_rate) / (1 - self.improvement_rate),
            "Loss Delta": self.loss_delta
        }

    def _print_status(self):
        for metric_name, value in self._generate_metrics().items():
            print(f"{metric_name}: ", value)
        print("")

    def _report_status(self, logger):
        for metric_name, value in self._generate_metrics().items():
            if metric_name == "Iterations":
                continue

            logger.report_scalar(
                metric_name, "Series", iteration=self.iterations, value=value
            )

    def get_cluster_map(self):
        return self.cluster_map.cluster_map


def loss(left_orders, right_orders, cluster_map):
    left_cluster_mask = left_orders @ cluster_map.T
    right_cluster_mask = right_orders @ cluster_map.T

    return np.minimum(left_cluster_mask, right_cluster_mask).sum()
