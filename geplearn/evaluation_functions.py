import numpy as np
from typing import Callable, List, Optional, Tuple, Union
from geppy import Chromosome, Toolbox
from numba import njit
from .utils import jaccard_similarity, cosine_distance, normalised_euclidean_distance


@njit
def perform_linear_scaling(
    y_true: np.ndarray, y_predicted: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Performs linear scaling.

    Parameters
    ----------
    y_true: np.ndarray
        True targets.
    y_predicted: np.ndarray
        Predicted targets.

    Returns
    -------
    Tuple[float, float, np.ndarray]
        Tuple with the slope, intercept, and scaled predicted targets.

    """
    Q = np.hstack((np.reshape(y_predicted, (-1, 1)), np.ones((len(y_predicted), 1))))

    items, _, _, _ = np.linalg.lstsq(Q, y_true)

    return items[0], items[1], items[0] * y_predicted + items[1]


@njit
def compute_rmse(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between the predicted and
    true targets.

    Parameters
    ----------
    y_true: np.ndarray
        True targets.
    y_predicted: np.ndarray
        Predicted targets.

    Returns
    -------
    rmse: float
        Root mean squared error.
    """
    rmse = np.sqrt(np.mean(y_true - y_predicted) ** 2)

    return rmse


def rmse(
    individual: Chromosome,
    toolbox: Toolbox,
    X: Union[np.ndarray, List],
    y: np.ndarray,
    linear_scaling: bool = True,
) -> Chromosome:
    """
    Computes the rmse between the true targets and the predicted
    targets.

    Parameters
    ----------
    individual: Chromosome
        Individual to compile and evaluate.
    toolbox: Toolbox
        GEP Toolbox containing the compile function.
    X: Union[np.ndarray, List]
        Features matrix.
    y: np.ndarray
        Target vector.
    linear_scaling: bool
        Whether to use linear scaling or not.

    Returns
    -------
    individual: Chromosome
        Updated individual with fitness value and linear scaling coefficients
        if linear_scaling==True.
    """
    func = toolbox.compile(individual)

    y_predicted = func(*X) if isinstance(X, list) else func(X)

    if not isinstance(y_predicted, np.ndarray):
        # y_predicted is a GPU array, we need to copy it to host.
        y_predicted = y_predicted.copy_to_host()

    if linear_scaling:

        a, b, y_predicted_scaled = perform_linear_scaling(y, y_predicted)

        individual.a = a

        individual.b = b

        score = (compute_rmse(y, y_predicted_scaled),)

    else:

        score = (compute_rmse(y, y_predicted),)

    individual.fitness.values = score

    return individual


def correlation(
    individual: Chromosome,
    toolbox: Toolbox,
    X: Union[np.ndarray, List],
    y: np.ndarray,
    linear_scaling: bool = True,
) -> Chromosome:
    """
    Computes the correlation between the true targets and the predicted
    targets.

    Parameters
    ----------
    individual: Chromosome
        Individual to compile and evaluate.
    toolbox: Toolbox
        GEP Toolbox containing the compile function.
    X: Union[np.ndarray, List]
        Features matrix.
    y: np.ndarray
        Target vector.
    linear_scaling: bool
        Whether to use linear scaling or not.

    Returns
    -------
    individual: Chromosome
        Updated individual with fitness value and linear scaling coefficients
        if linear_scaling==True.
    """
    func = toolbox.compile(individual)

    y_predicted = func(*X) if isinstance(X, list) else func(X)

    if not isinstance(y_predicted, np.ndarray):
        # y_predicted is a GPU array, we need to copy it to host.

        y_predicted = y_predicted.copy_to_host()

    if linear_scaling:

        a, b, y_predicted_scaled = perform_linear_scaling(y, y_predicted)

        individual.a = a

        individual.b = b

        score = (np.corrcoef([y, y_predicted_scaled])[0, 1],)

    else:

        score = (np.corrcoef([y, y_predicted])[0, 1],)

    individual.fitness.values = score

    return individual


def sklearn_evaluation(
    individual: Chromosome,
    toolbox: Toolbox,
    X: Union[np.ndarray, List],
    y: np.ndarray,
    sk_metric: Callable,
    linear_scaling: bool = True,
) -> Chromosome:
    """
    Evaluates the individual using a sklearn compatible metric.

    Parameters
    ----------
    individual: Chromosome
        Individual to compile and evaluate.
    toolbox: Toolbox
        GEP Toolbox containing the compile function.
    X: Union[np.ndarray, List]
        Features matrix.
    y: np.ndarray
        Target vector.
    sk_metric: Callable
        Function to assess the fitness of an individual. It must have the same
        structure as scikit learn metrics (y_true and y_pred as inputs).
    linear_scaling: bool
        Whether to use linear scaling or not.

    Returns
    -------
    individual: Chromosome
        Updated individual with fitness value and linear scaling coefficients
        if linear_scaling==True.
    """

    func = toolbox.compile(individual)

    y_predicted = func(*X) if isinstance(X, list) else func(X)

    if not isinstance(y_predicted, np.ndarray):
        # y_predicted is a GPU array, we need to copy it to host.
        y_predicted = y_predicted.copy_to_host()

    if linear_scaling:

        a, b, y_predicted_scaled = perform_linear_scaling(y, y_predicted)
        individual.a = a
        individual.b = b
        score = (sk_metric(y, y_predicted_scaled),)

    else:

        score = (sk_metric(y, y_predicted),)

    individual.fitness.values = score

    return individual


def counterfactuals_evaluation(
    individuals: List[Chromosome],
    toolbox: Toolbox,
    predict_proba_fn: Callable,
    X_obs: np.ndarray,
    distance: str,
    predicted_class: int,
    cat_columns: Optional[List[int]] = None,
    cont_columns: Optional[List[int]] = None,
    weight_cat: float = 0.5,
    weight_cont: float = 0.5,
    threshold: float = 0.5,
    weight_input_loss: float = 0.5,
    weight_prediction_loss: float = 0.5,
) -> np.ndarray:
    """
    It evaluates the fitness of the conterfactual examples. The fitness
    function is composed of two parts:
        - Similarity of the counterfactual example with the actual observation.
        - Difference in the predicted output.
    The idea is to create a synthetic observation that is as similar as
    possible to the original one, but the model prediction on the synthetic
    observation is the opposite (without changing too much the output
    probability). For instance, if the model uses a threshold of 0.5, and
    the predicted class for the original observation is 1, we want the
    predicted probability for the synthetic observation to be 0.49 (class 0).

    Parameters
    ----------
    individuals: List[Chromosome]
        List of individuals (synthetic observations).
    toolbox: Toolbox
        GEP Toolbox.
    predict_proba_fn: Callable
        Function that predicts the probability of an observation being of class
        0 or 1.
    X_obs: np.ndarray
        Observation we want to explain.
    distance: str
        Distance metric to use for continuous variables. It can be 'cosine' or
        'ned' (normalized euclidean distance).
    predicted_class: int
        Class predicted by the model for the original observation.
    cat_columns: List[int]
        List of categorical columns indexes.
    cont_columns: List[int]
        List of continuous columns indexes.
    weight_cat: float
        Weight for the similarity measure between categorical features of the
        synthetic observation and the original observation.
    weight_cont: float
        Weight for the similarity measure between continuous features of the
        synthetic observation and the original observation.
    threshold: float
        Threshold value used by the model to decide a class from the probability
        value.
    weight_input_loss: float
        Weight for the part of the loss function related to the similarity
        between the features' values.
    weight_prediction_loss: float
        Weight for the part of the loss function related to the similarity
        between the predicted probabilities.

    Returns
    -------
    scores: np.ndarray
        Scores of the counterfactual examples.
    """
    if not cat_columns and not cont_columns:
        raise ValueError(
            "At least one of 'cat_columns' or 'cont_columns' must be provided."
        )

    X_synth = np.array([toolbox.compile(ind) for ind in individuals])

    synth_probs = predict_proba_fn(X_synth)[:, predicted_class]

    js = (
        jaccard_similarity(X_synth[:, cat_columns], X_obs[cat_columns])
        if cat_columns
        else 0.0
    )

    if distance == "cosine":

        dist = (
            cosine_distance(X_synth[:, cont_columns], X_obs[cont_columns])
            if cont_columns
            else 0.0
        )

    elif distance == "ned":

        dist = (
            normalised_euclidean_distance(X_synth[:, cont_columns], X_obs[cont_columns])
            if cont_columns
            else 0.0
        )

    loss_inputs = weight_cat * js + weight_cont * dist

    loss_pred = np.where(synth_probs < threshold, 1.0 - (threshold - synth_probs), 0.0)

    scores = weight_input_loss * loss_inputs + weight_prediction_loss * loss_pred

    return scores
