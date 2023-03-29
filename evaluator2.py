from typing import NamedTuple, Optional
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

class EvalResult(NamedTuple):
    recall: NDArray[np.floating]
    ndcg: NDArray[np.floating]
    hitrate: NDArray[np.floating]
    precision: NDArray[np.floating]

class Evaluator(object):
    def __init__(self, train, test, k=20):
        """
        Create a evaluator for recall@K evaluation
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the calculations
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.train_matrix = train
        self.test_matrix = test
        self.k = k

    def eval(self,
        user_scores: NDArray[np.floating],
        user_idxs: NDArray[np.integer]
    ) -> EvalResult:
        train = self.train_matrix[user_idxs, :].toarray()
        labels = self.test_matrix[user_idxs, :].toarray()

        # Set user scores in the train set as -inf to avoid evaluating the
        # model on items it was trained on.
        # Alternatively we can set the labels for the items in the test set
        # to zero
        fixed_scores = np.where(train == 0, user_scores, -np.inf)
        top_scores_idxs = find_ktop(fixed_scores, self.k)

        common_args = {
            # Ignore logits for training labels
            'scores': fixed_scores,
            'labels': labels,
            'k': self.k,
            'top_scores_idxs': top_scores_idxs
        }

        return EvalResult(
            recall=recall_at_k(**common_args),
            ndcg=ndcg_at_k(**common_args),
            hitrate=hitrate_at_k(**common_args),
            precision=precision_at_k(**common_args)
        )

def find_ktop(a: NDArray, k: int):
    return np.argpartition(a, axis=1, kth=-k)[:, -k-1:-1]

def measure_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int,
    top_scores_idxs: Optional[NDArray[np.floating]] = None,
) -> float:
    assert scores.shape == labels.shape

    if top_scores_idxs is None:
        top_scores_idxs = find_ktop(scores, k)

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
    k: int,
    top_scores_idxs: Optional[NDArray[np.floating]] = None,
) -> float:
    true_positives, relevant,  _ = measure_at_k(scores, labels, k, top_scores_idxs)
    return true_positives / np.maximum(relevant, 1)


def hitrate_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int,
    top_scores_idxs: Optional[NDArray[np.floating]] = None,
) -> float:
    true_positives, *_  = measure_at_k(scores, labels, k, top_scores_idxs)
    return (true_positives > 0).astype(float)


def precision_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int,
    top_scores_idxs: Optional[NDArray[np.floating]] = None,
) -> float:
    true_positives, _,  retrieved = measure_at_k(scores, labels, k, top_scores_idxs)
    return true_positives / np.maximum(retrieved, 1)


def ndcg_at_k(
    scores: NDArray[np.floating],
    labels: NDArray[np.floating],
    k: int,
    top_scores_idxs: Optional[NDArray[np.floating]] = None,
) -> float:
    assert scores.shape == labels.shape

    binarized_labels = (labels > 0)

    if top_scores_idxs is None:
        top_scores_idxs = find_ktop(scores, k)

    top_labels_idxs = find_ktop(labels, k)

    row_idxs, _ = np.indices(top_scores_idxs.shape)

    true_positives = binarized_labels[row_idxs, top_scores_idxs]
    top_labels = binarized_labels[row_idxs, top_labels_idxs]

    denominator = np.log2(2 + np.arange(k))
    dcg = np.sum(true_positives / denominator, axis=1)
    idcg = np.maximum(np.sum(top_labels / denominator, axis=1), 1e-9)
    return dcg / idcg
