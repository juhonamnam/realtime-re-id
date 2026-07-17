from .common import *
from .dist_func.dist_mean import DistMean
from .dist_func.concat_dist import ConcatDist

__all__ = ['DistanceMetric']

DISTANCE_METRIC_REGISTRY = {
    'concat_dist': ConcatDist(),
    'dist_mean': DistMean(),
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

    def default(self, feature1, feature2, *args, **kwargs):
        return self.metric(self.default_metric, feature1, feature2, *args, **kwargs)

    def metric(self, metric_name, feature1, feature2, return_part_distances=False, return_v_scores=False, cross_batch=False):
        if metric_name not in self.metric_dict:
            raise ValueError(
                f"Unsupported distance metric method: {metric_name}")
        distance_func = self.metric_dict[metric_name]

        if cross_batch:
            v_scores, part_distances = prepare_for_distance_metric_cross_batch(
                feature1, feature2)
            final_distance = distance_func.cross_batch(
                v_scores, part_distances)
        else:
            v_scores, part_distances = prepare_for_distance_metric(
                feature1, feature2)
            final_distance = distance_func.single(v_scores, part_distances)

        returning = [final_distance]

        if return_part_distances:
            returning.append(part_distances)
        if return_v_scores:
            returning.append(v_scores)
        return returning if len(returning) > 1 else returning[0]

    def all(self, feature1, feature2, cross_batch=False):
        if cross_batch:
            v_scores, part_distances = prepare_for_distance_metric_cross_batch(
                feature1, feature2)
        else:
            v_scores, part_distances = prepare_for_distance_metric(
                feature1, feature2)

        final_distances = []

        for metric in self.metric_list:
            if cross_batch:
                final_distance = metric.cross_batch(v_scores, part_distances)
            else:
                final_distance = metric.single(v_scores, part_distances)

            final_distances.append(final_distance)

        return final_distances, part_distances

    def __call__(self, feature1, feature2, *args, **kwargs):
        return self.default(feature1, feature2, *args, **kwargs)
