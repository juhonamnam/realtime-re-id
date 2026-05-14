import torch

__all__ = ['Person', 'cluster_people']

class Person:
    def __init__(self, v_scores: torch.Tensor, emb_vecs: torch.Tensor):
        self.v_scores = v_scores
        self.emb_vecs = emb_vecs
        self.v_scores_sum = v_scores.sum().item()

    def __str__(self):
        return f"Person(v_scores={self.v_scores}, emb_vecs={self.emb_vecs})"

class Cluster(Person):
    def __init__(self, person: Person):
        super().__init__(person.v_scores, person.emb_vecs)
        self.members = [person]

    def add_member(self, person: Person):
        self.members.append(person)
        for i in range(len(self.v_scores)):
            if person.v_scores[i] > self.v_scores[i]:
                self.v_scores[i] = person.v_scores[i]
                self.emb_vecs[i] = person.emb_vecs[i]

def get_comparability(query: Person, cluster: Cluster) -> float:
    comparability = 0.0
    for i in range(len(query.v_scores)):
        comparability += min(query.v_scores[i], cluster.v_scores[i]).item()
    return comparability

def cluster_people(people: list[Person], threshold: float, similarity_func: callable):
    labels = [0] * len(people)
    clusters = []
    sorted_people = sorted(map(lambda idx, person: (idx, person), range(len(people)), people),
                           key=lambda x: x[1].v_scores_sum, reverse=True)
    for person_idx, person in sorted_people:
        best_cluster = None
        best_sim_x_comp = -1
        best_cluster_index = -1

        for cluster_idx, cluster in enumerate(clusters):
            similarity = similarity_func(person, cluster)
            if similarity < threshold:
                continue
            comparability = get_comparability(person, cluster)
            sim_x_comp = similarity * comparability
            if sim_x_comp > best_sim_x_comp:
                best_sim_x_comp = sim_x_comp
                best_cluster = cluster
                best_cluster_index = cluster_idx

        if best_cluster is not None:
            best_cluster.add_member(person)
            labels[person_idx] = best_cluster_index
        else:
            new_cluster = Cluster(person)
            clusters.append(new_cluster)
            labels[person_idx] = len(clusters) - 1

    return labels