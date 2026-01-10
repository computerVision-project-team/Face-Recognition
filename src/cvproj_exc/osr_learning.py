from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd

from cvproj_exc.config import Config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans

UNKNOWN_LABEL: Final[int] = -1
def _make_backbone(random_state: int = 0) -> Pipeline:
    """
    Fast and solid multiclass backbone for many classes.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    multi_class="auto",
                    max_iter=2000,
                    C=4.0,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _known_score_from_proba(proba: np.ndarray, classes: np.ndarray, kc_mask: np.ndarray) -> np.ndarray:
    """
    known_score = max probability over KC classes only
    """
    if not np.any(kc_mask):
        return np.max(proba, axis=1)
    return np.max(proba[:, kc_mask], axis=1)


def _calibrate_threshold_from_kuc(
    x_train: np.ndarray,
    y_train_pseudo: np.ndarray,
    proba_fn,
    classes: np.ndarray,
    kc_mask: np.ndarray,
    pseudo_mask: np.ndarray,
    far_target: float = 0.01,
) -> float:
    """
    Choose threshold so that about far_target of KUC are (wrongly) accepted as known.

    We compute known_score on KUC samples and set threshold to the (1 - far_target) percentile.
    Example: far_target=1% => threshold = 99th percentile of known_scores on KUC.
    """
    kuc_idx = np.where(pseudo_mask)[0]
    if kuc_idx.size == 0:
        return 0.5

    proba_kuc = proba_fn(x_train[kuc_idx])
    known_score_kuc = _known_score_from_proba(proba_kuc, classes, kc_mask)

    p = 100.0 * (1.0 - far_target)  # 99 for 1% FAR
    tau = float(np.percentile(known_score_kuc, p))
    return float(np.clip(tau, 0.05, 0.95))

def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'spl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)

    kc_labels = np.unique(y_train[y_train >= 0])
    if kc_labels.size == 0:
        raise ValueError("No KC labels (>=0) found in y_train.")

    # pseudo unknown class label u
    u = int(kc_labels.max() + 1)

    y_train_pseudo = y_train.copy()
    y_train_pseudo[y_train_pseudo == UNKNOWN_LABEL] = u

    backbone = _make_backbone(random_state=0)
    backbone.fit(x_train, y_train_pseudo)

    classes = backbone.named_steps["clf"].classes_
    kc_mask = np.isin(classes, kc_labels)
    pseudo_mask_train = y_train_pseudo == u

    def proba_fn(xx: np.ndarray) -> np.ndarray:
        return backbone.predict_proba(xx)

    # Calibrate threshold using KUC samples (~1% FAR on KUC-like unknowns)
    tau = _calibrate_threshold_from_kuc(
        x_train=x_train,
        y_train_pseudo=y_train_pseudo,
        proba_fn=proba_fn,
        classes=classes,
        kc_mask=kc_mask,
        pseudo_mask=pseudo_mask_train,
        far_target=0.01,
    )
    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.
        x_test = np.asarray(x_test, dtype=np.float32)
        proba = backbone.predict_proba(x_test)  # (n, C)
        best_idx = np.argmax(proba, axis=1)
        best_label = classes[best_idx].astype(int)

        known_score = _known_score_from_proba(proba, classes, kc_mask)  # (n,)

        # Decide known vs unknown:
        # - if predicted pseudo label u -> unknown
        # - or if KC confidence too low -> unknown
        is_pred_u = best_label == u
        is_low_conf = known_score < tau
        is_unknown = is_pred_u | is_low_conf

        y_pred = best_label.copy()
        y_pred[is_unknown] = UNKNOWN_LABEL

        # Score should be "confidence of the predicted label"
        # - if known: higher known_score => more confident
        # - if unknown: higher (1-known_score) => more confident
        y_score = known_score.copy()
        y_score[is_unknown] = 1.0 - known_score[is_unknown]

        return y_pred, y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'mpl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)

    kc_labels = np.unique(y_train[y_train >= 0])
    if kc_labels.size == 0:
        raise ValueError("No KC labels (>=0) found in y_train.")

    kuc_mask = y_train == UNKNOWN_LABEL
    x_kuc = x_train[kuc_mask]

    # If too few KUC samples, fall back to SPL behavior (still valid)
    if x_kuc.shape[0] < 10:
        return spl_training(x_train, y_train)

    # Number of pseudo classes for KUC (keep moderate for speed)
    n_pseudo = min(50, x_kuc.shape[0])  # up to 50 clusters
    n_pseudo = max(2, n_pseudo)

    kmeans = MiniBatchKMeans(
        n_clusters=n_pseudo,
        random_state=0,
        batch_size=1024,
        n_init="auto",
        max_iter=200,
    )
    kuc_cluster = kmeans.fit_predict(x_kuc)

    start = int(kc_labels.max() + 1)
    pseudo_labels = start + np.arange(n_pseudo, dtype=int)

    y_train_pseudo = y_train.copy()
    y_train_pseudo[kuc_mask] = pseudo_labels[kuc_cluster]

    backbone = _make_backbone(random_state=0)
    backbone.fit(x_train, y_train_pseudo)

    classes = backbone.named_steps["clf"].classes_
    kc_mask_cls = np.isin(classes, kc_labels)

    # training-time mask for "pseudo samples"
    pseudo_mask_train = np.isin(y_train_pseudo, pseudo_labels)

    def proba_fn(xx: np.ndarray) -> np.ndarray:
        return backbone.predict_proba(xx)

    # Calibrate threshold using KUC samples (~1% FAR on KUC-like unknowns)
    tau = _calibrate_threshold_from_kuc(
        x_train=x_train,
        y_train_pseudo=y_train_pseudo,
        proba_fn=proba_fn,
        classes=classes,
        kc_mask=kc_mask_cls,
        pseudo_mask=pseudo_mask_train,
        far_target=0.01,
    )
    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = np.asarray(x_test, dtype=np.float32)
        proba = backbone.predict_proba(x_test)
        best_idx = np.argmax(proba, axis=1)
        best_label = classes[best_idx].astype(int)

        known_score = _known_score_from_proba(proba, classes, kc_mask_cls)

        is_pred_pseudo = np.isin(best_label, pseudo_labels)
        is_low_conf = known_score < tau
        is_unknown = is_pred_pseudo | is_low_conf

        y_pred = best_label.copy()
        y_pred[is_unknown] = UNKNOWN_LABEL

        y_score = known_score.copy()
        y_score[is_unknown] = 1.0 - known_score[is_unknown]

        return y_pred, y_score


    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()
    spl_predict_fn = spl_training(x_train, y_train)
    mpl_predict_fn = mpl_training(x_train, y_train)

    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, y_score = predict_fn(x_test)
        print("Acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == "__main__":
    main()
