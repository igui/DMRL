from time import perf_counter
import numpy as np
from numpy.typing import NDArray

class Evaluator(object):
    def __init__(self, model, train, test, k=20):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the calculations
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_matrix = train.toarray()
        self.test_matrix = test.toarray()
        self.k = k


    def eval(self, sess, users):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: hitratio,ndgg@K
        """
        t1 = perf_counter()
        user_scores,_scores_s,_scores_w = sess.run(self.model.item_scores,
                                {self.model.score_user_ids: users})
        t2 = perf_counter()

        # Ignore logits for training labels
        train = self.train_matrix[users, :]
        labels = self.test_matrix[users, :]

        fixed_scores = np.where(train == 0, user_scores, -np.inf)
        common_args = { 'scores': fixed_scores, 'labels': labels, 'k': self.k }

        recalls = recall_at_k(**common_args)
        ndcgs = ndcg_at_k(**common_args)
        hitrates = hitrate_at_k(**common_args)
        precisions = precision_at_k(**common_args)
        t3 = perf_counter()

        #print(f'Model Eval {t2 - t1:.2f}s Metric Eval {t3-t2:.2f}s')

        return recalls, ndcgs, hitrates, precisions


def measure_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int
) -> float:
    assert scores.shape == labels.shape

    top_scores_idxs = np.fliplr(np.argsort(scores, axis=1))[:, :k]

    binarized_labels = (labels > 0)
    binarized_scores = (scores > 0)

    row_idxs, _ = np.indices(top_scores_idxs.shape)

    true_positives = np.sum(
        binarized_labels[row_idxs, top_scores_idxs],
        axis=1
    )
    relevant = np.minimum(binarized_labels.sum(axis=1), k)
    retrieved = np.minimum(binarized_scores.sum(axis=1), k)
    return true_positives, relevant, retrieved


def recall_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int
) -> float:
    true_positives, relevant,  _ = measure_at_k(scores, labels, k)
    return true_positives / np.maximum(relevant, 1)


def hitrate_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int
) -> float:
    true_positives_sum, *_  = measure_at_k(scores, labels, k)
    return (true_positives_sum > 0).astype(float)


def precision_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int
) -> float:
    true_positives, _,  retrieved = measure_at_k(scores, labels, k)
    return true_positives / np.maximum(retrieved, 1)


def ndcg_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int
) -> float:
    assert scores.shape == labels.shape

    binarized_labels = (labels > 0)
    top_scores_idxs = np.fliplr(np.argsort(scores, axis=1))[:, :k]
    top_labels_idxs = np.fliplr(np.argsort(binarized_labels, axis=1))[:, :k]

    row_idxs, _ = np.indices(top_scores_idxs.shape)

    true_positives = binarized_labels[row_idxs, top_scores_idxs]
    top_labels = binarized_labels[row_idxs, top_labels_idxs]

    denominator = np.log2(2 + np.arange(k))
    dcg = np.sum(true_positives / denominator, axis=1)
    idcg = np.maximum(np.sum(top_labels / denominator, axis=1), 1e-9)
    return dcg / idcg
