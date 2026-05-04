import torch.nn as nn

from .common import *
from .weakest_link import get_weakest_link_similarity_score
from .weighted_mean import get_weighted_mean_similarity_score
from .weighted_geometric_mean import get_weighted_geometric_mean_similarity_score
from .minimum import get_minimum_similarity_score

__all__ = ['SimilarityMetric']

SIMILARITY_METRIC_REGISTRY = {
    'weakest_link': get_weakest_link_similarity_score,
    'weighted_mean': get_weighted_mean_similarity_score,
    'weighted_geometric_mean': get_weighted_geometric_mean_similarity_score,
    'minimum': get_minimum_similarity_score,
}

SIMILARITY_METRICS = list(SIMILARITY_METRIC_REGISTRY.keys())

class SimilarityMetric():
    similarity_metrics = SIMILARITY_METRICS
    metric_num = len(similarity_metrics)
    def __init__(self, default_method='weakest_link', default_part_thresholds=0.5):
        super(SimilarityMetric, self).__init__()
        if default_method in SIMILARITY_METRIC_REGISTRY:
            self.default_similarity_func = SIMILARITY_METRIC_REGISTRY[default_method]
        else:
            raise ValueError(f"Unsupported similarity metric method: {default_method}")
        self.default_part_thresholds = default_part_thresholds

        self.metric_list = []
        self.metric_dict = {}
        for method in self.similarity_metrics:
            self.metric_list.append(SIMILARITY_METRIC_REGISTRY[method])
            self.metric_dict[method] = SIMILARITY_METRIC_REGISTRY[method]

    def default(self, feature1, feature2, part_thresholds=None, return_part_scores=False):
        if part_thresholds is None:
            part_thresholds = self.default_part_thresholds
        v_scores, part_s_scores = prepare_for_similarity_metric(feature1, feature2)
        
        total_s_score = self.default_similarity_func(v_scores, part_s_scores, part_thresholds)
        if return_part_scores:
            return total_s_score, part_s_scores
        return total_s_score

    def metric(self, metric_name, feature1, feature2, part_thresholds=None):
        if part_thresholds is None:
            part_thresholds = self.default_part_thresholds
        if metric_name not in self.metric_dict:
            raise ValueError(f"Unsupported similarity metric method: {metric_name}")
        similarity_func = self.metric_dict[metric_name]
        v_scores, part_s_scores = prepare_for_similarity_metric(feature1, feature2)
        total_s_score = similarity_func(v_scores, part_s_scores, part_thresholds)
        return total_s_score

    def all(self, feature1, feature2, part_thresholds=None):
        if part_thresholds is None:
            part_thresholds = self.default_part_thresholds
        v_scores, part_s_scores = prepare_for_similarity_metric(feature1, feature2)

        total_s_scores = []

        for metric in self.metric_list:
            total_s_score = metric(v_scores, part_s_scores, part_thresholds)
                
            total_s_scores.append(total_s_score)

        return total_s_scores, part_s_scores

    def __call__(self, feature1, feature2, part_thresholds=None, return_part_scores=False):
        return self.default(feature1, feature2, part_thresholds, return_part_scores)
