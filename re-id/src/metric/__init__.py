import torch.nn as nn

from .common import *
from .weakest_link import get_weakest_link_similarity_score
from .weighted_mean import get_weighted_mean_similarity_score
from .weighted_geometric_mean import get_weighted_geometric_mean_similarity_score
from .minimum import get_minimum_similarity_score

SIMILARITY_METRIC_REGISTRY = {
    'weakest_link': {"func": get_weakest_link_similarity_score,
                     "default_threshold": 0.64},
    'weighted_mean': {"func": get_weighted_mean_similarity_score,
                      "default_threshold": 0.66},
    'weighted_geometric_mean': {"func": get_weighted_geometric_mean_similarity_score,
                                "default_threshold": 0.64},
    'minimum': {"func": get_minimum_similarity_score,
                "default_threshold": 0.64}
}

SIMILARITY_METRICS = list(SIMILARITY_METRIC_REGISTRY.keys())

class SimilarityMetric(nn.Module):
    def __init__(self, method='weakest_link', thresholds=0.5):
        super(SimilarityMetric, self).__init__()
        if method in SIMILARITY_METRIC_REGISTRY:
            self.similarity_func = SIMILARITY_METRIC_REGISTRY[method]["func"]
        else:
            raise ValueError(f"Unsupported similarity metric method: {method}")
        self.thresholds = thresholds

    def forward(self, feature1, feature2, thresholds=None, return_part_scores=False):
        v_scores, part_s_scores = prepare_for_similarity_metric(feature1, feature2)
        if thresholds is None:
            thresholds = self.thresholds
        
        total_s_score = self.similarity_func(v_scores, part_s_scores, thresholds)
        if return_part_scores:
            return total_s_score, part_s_scores
        return total_s_score

class AllSimilarityMetrics(nn.Module):
    similarity_metrics = SIMILARITY_METRICS
    default_metric_thresholds = [SIMILARITY_METRIC_REGISTRY[method]["default_threshold"] for method in similarity_metrics]
    metric_num = len(similarity_metrics)
    def __init__(self, thresholds=0.5):
        super(AllSimilarityMetrics, self).__init__()
        self.metrics = []
        for method in self.similarity_metrics:
            self.metrics.append(SIMILARITY_METRIC_REGISTRY[method]["func"])
        self.thresholds = thresholds

    def forward(self, feature1, feature2, thresholds=None):
        v_scores, part_s_scores = prepare_for_similarity_metric(feature1, feature2)
        if thresholds is None:
            thresholds = self.thresholds

        total_s_scores = []

        for metric in self.metrics:
            total_s_score = metric(v_scores, part_s_scores, thresholds)
                
            total_s_scores.append(total_s_score)

        return total_s_scores, part_s_scores
