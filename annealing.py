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
            cluster_state_reg=0,
            split_value=0.5,
            bin_delta=1
    ):
        self.cluster_map = ClusterMap(data, clusters, bin_delta)
        self.data_count = data.shape[0]
        self.data_sum = data.sum(0).A.flatten()
        self.split_value = split_value
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
                merge_prob=merge_prob,
                split_value=split_value
            )
        else:
            self.change_generator = simple_change_generator(data, double_change_weight)

        self.improvement_rate = 0
        self.acceptance_rate = 0
        self.loss_delta = 0

        self.cummulative_base_loss = 0
        self.cummulative_reg = 0
        self.cummulative_state_comp = 0

        self.iterations = 0

        self.cluster_state_reg = cluster_state_reg

        self.base_loss = (
            loss(self.valid_orders[0], self.valid_orders[1], self.cluster_map.cluster_map)
            / self.valid_orders[0].shape[0]
        )

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
        state_reg = self._state_change_loss(state_changes)
        loss_comp = self._self_conflict_loss_compensation(state_changes)
        total_exp_change = state_reg + exp_change + loss_comp

        if total_exp_change < 0 or np.exp(-total_exp_change / self.T) > np.random.rand():
            for matrix_change in matrix_changes:
                self.cluster_map.apply_change(matrix_change)
            for state_change in state_changes:
                self.cluster_map.apply_state_change(state_change)
                self.state_sum += state_change.delta
            self.acceptance_rate += 0.005
            self.improvement_rate += 0.005 * (total_exp_change < 0)
            self.loss_delta += 0.005 * total_exp_change

            self.cummulative_base_loss += exp_change / (self.data_count ** 2)
            self.cummulative_reg += state_reg / (self.data_count ** 2)
            self.cummulative_state_comp += loss_comp / (self.data_count ** 2)

        self.T *= self.decay

    def _state_change_loss(self, state_changes):
        state_change = sum([change.delta for change in state_changes])
        return (
            np.exp((self.state_sum + state_change) * self.cluster_state_reg)
            - np.exp(self.state_sum * self.cluster_state_reg)
        )

    def _self_conflict_loss_compensation(self, state_changes):
        comp = 0
        for state_change in state_changes:
            comp += (
                (self.data_sum[state_change.sku_id] ** 2)
                * (1 - self.split_value ** 2)
                * state_change.delta
            )
        return comp

    def anneal(self, iters, verbose=0, logger=None):
        for _ in range(iters):
            self.anneal_once()
            if self.iterations % 1000 == 0:
                if verbose:
                    self._print_status()
                if logger is not None:
                    self._report_status(logger)

    def _generate_metrics(self):
        sample_loss = (loss(self.valid_orders[0], self.valid_orders[1], self.cluster_map.cluster_map)
                       / self.valid_orders[0].shape[0])
        return {
            "Iterations": self.iterations,
            "Sample Loss": sample_loss,
            "Loss Discrepancy": self.base_loss + self.cummulative_base_loss - sample_loss,
            "Total Loss": self.base_loss + self.cummulative_base_loss + self.cummulative_reg + self.cummulative_state_comp,
            "No Reg Loss": self.base_loss + self.cummulative_base_loss + self.cummulative_state_comp,
            "Improvement Rate": self.improvement_rate,
            "Acceptance Rate": (self.acceptance_rate - self.improvement_rate) / (1 - self.improvement_rate),
            "Loss Delta": self.loss_delta,
            "Splitted Items": self.state_sum
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
