from algorithms.Processor import Processor

from sklearn.metrics import silhouette_score

def eval_cluster(cluster: Processor) -> float:

    X, tags = cluster.get_all_instances_with_tags()
    labels = []
    labels_set = set()

    for tag in tags:
        label = cluster.get_cluster_by_tag(tag)
        labels.append(label)
        labels_set.add(label)

    try:

        Ss = silhouette_score(X, labels)
    except:

        Ss = -1.0

    return Ss