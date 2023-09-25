import numpy as np

from changes import MatrixChange, StateChange, simple_change_generator, split_change_generator
from cluster_map import ClusterMap
from data_load import sample_data


class Annealing:
    def __init__(
            self,
            clusters, T, decay,
            valid_sample_count,
            data,
            use_split_clusters=False,
            double_change_weight=0,
            split_weight=1,
            merge_prob=0.3,
            initial_split_proportion=0,
            cluster_state_reg=0
    ):
        self.cluster_map = ClusterMap(data, clusters, initial_split_proportion)
        self.state_sum = sum(self.cluster_map.state_map)
        self.T = T
        self.decay = decay
        self.valid_orders = (
            sample_data(data, valid_sample_count),
            sample_data(data, valid_sample_count)
        )

        if use_split_clusters:
            self.change_generator = split_change_generator(
                data=data,
                double_sugggestion_weight=double_change_weight,
                split_weight=split_weight,
                merge_prob=merge_prob
            )
        else:
            self.change_generator = simple_change_generator(data, double_change_weight)

        self.improvement_rate = 0
        self.acceptance_rate = 0
        self.loss_delta = 0
        self.iterations = 0

        self.cluster_state_reg = cluster_state_reg

    def anneal_once(self):
        self.acceptance_rate *= 0.995
        self.improvement_rate *= 0.995
        self.loss_delta *= 0.995
        self.iterations += 1

        matrix_changes, state_changes = self.change_generator.suggest_changes(
            cluster_map=self.cluster_map.cluster_map,
            cluster_state=self.cluster_map.state_map
        )
        exp_change = self.cluster_map.calculate_loss(matrix_changes)
        exp_change += self._state_change_loss(state_changes)

        if exp_change < 0 or np.exp(-exp_change / self.T) > np.random.rand():
            for matrix_change in matrix_changes:
                self.cluster_map.apply_change(matrix_change)
            for state_change in state_changes:
                self.cluster_map.apply_state_change(state_change)
                self.state_sum += state_change.delta
            self.acceptance_rate += 0.005
            self.improvement_rate += 0.005 * (exp_change < 0)
            self.loss_delta += 0.005 * exp_change

        self.T *= self.decay

    def _state_change_loss(self, state_changes):
        state_change = sum([change.delta for change in state_changes])
        return np.exp(self.state_sum * self.cluster_state_reg) * self.cluster_state_reg * state_change

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
