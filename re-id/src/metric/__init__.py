from .common import *
from .dist_sum import get_dist_sum
from .dist_mean import get_dist_mean
from .dist_max import get_dist_max
from .concat_dist import get_concat_dist

__all__ = ['DistanceMetric']

DISTANCE_METRIC_REGISTRY = {
    'concat_dist': get_concat_dist,
    'dist_mean': get_dist_mean,
    'dist_sum': get_dist_sum,
    'dist_max': get_dist_max,
}

DISTANCE_METRICS = list(DISTANCE_METRIC_REGISTRY.keys())


class DistanceMetric():
    distance_metrics = DISTANCE_METRICS
    metric_num = len(distance_metrics)

    def __init__(self, default_metric='concat_dist'):
        super(DistanceMetric, self).__init__()
        if default_metric in DISTANCE_METRIC_REGISTRY:
            self.default_metric = default_metric
        else:
            raise ValueError(
                f"Unsupported distance metric method: {default_metric}")

        self.metric_list = []
        self.metric_dict = {}
        for method in self.distance_metrics:
            self.metric_list.append(DISTANCE_METRIC_REGISTRY[method])
            self.metric_dict[method] = DISTANCE_METRIC_REGISTRY[method]

    def default(self, feature1, feature2, return_part_distances=False, return_v_scores=False):
        return self.metric(self.default_metric, feature1, feature2, return_part_distances=return_part_distances,
                           return_v_scores=return_v_scores)

    def metric(self, metric_name, feature1, feature2, return_part_distances=False, return_v_scores=False):
        if metric_name not in self.metric_dict:
            raise ValueError(
                f"Unsupported distance metric method: {metric_name}")
        distance_func = self.metric_dict[metric_name]
        v_scores, part_distances = prepare_for_distance_metric(
            feature1, feature2)
        final_distance = distance_func(v_scores, part_distances)

        returning = [final_distance]

        if return_part_distances:
            returning.append(part_distances)
        if return_v_scores:
            returning.append(v_scores)
        return returning if len(returning) > 1 else returning[0]

    def all(self, feature1, feature2):
        v_scores, part_distances = prepare_for_distance_metric(
            feature1, feature2)

        final_distances = []

        for metric in self.metric_list:
            final_distance = metric(v_scores, part_distances)

            final_distances.append(final_distance)

        return final_distances, part_distances

    def __call__(self, feature1, feature2, return_part_distances=False, return_v_scores=False):
        return self.default(feature1, feature2, return_part_distances=return_part_distances,
                            return_v_scores=return_v_scores)
