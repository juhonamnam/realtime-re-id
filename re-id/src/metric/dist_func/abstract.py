import torch


class DistanceFunction:
    MIN_V_SCORE = 0.1

    def single(self, v_scores: torch.Tensor, part_distances: torch.Tensor) -> torch.Tensor:
        """Compute the distance between two features.

        Args:
            v_scores (torch.Tensor): Visibility scores of the features.
            part_distances (torch.Tensor): Part distances of the features.

        Returns:
            torch.Tensor: Distance between the two features.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def cross_batch(self, v_scores: torch.Tensor, part_distances: torch.Tensor) -> torch.Tensor:
        """Compute the distance between two batches of features.

        Args:
            v_scores (torch.Tensor): Visibility scores of the features.
            part_distances (torch.Tensor): Part distances of the features.

        Returns:
            torch.Tensor: Distance matrix between the two batches of features.
        """
        raise NotImplementedError("Subclasses must implement this method.")
