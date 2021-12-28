from dataclasses import dataclass, field
import numpy as np
from typing import Callable, List, Tuple, Dict
import warnings
from .gep_core import (
    CounterfactualGene,
    counterfactual_compile,
    FitnessMax,
    gep_counterfactual,
)
from geppy import Chromosome, Toolbox
from deap.tools import selTournament, HallOfFame
from copy import copy
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from tqdm import tqdm


@dataclass
class CounterfactualExplainer:
    """
    Class for performing counterfactual explanation.

    Parameters
    ----------
    predict_proba_fn: Callable
        Function that predicts the probability on observations.
    X_obs: np.ndarray
        Observation to explain.
    threshold: float
        Threhsold value used by the model to define the predicted class.
    features_names: List[str]
        Names of the features.
    categorical_features_index: List[int]
        List of categorical features indexes.
    continuous_features_index: List[int]
        List of continuous features indexes.
    count_features_index: List[int]
        List of count features indexes.
    categorical_features_values: Dict[int, Tuple]
        Dictionary of unique values for each categorical feature. Keys are the
        indexes of the categorical features, values are their unique values.
    continuous_features_range: Dict[int, Tuple]
        Dictionary of the ranges of the continuous features. Keys are the
        indexes of the continuous features, values are a tuple of minimum and
        maximum values.
    count_features_range: Dict[int, Tuple]
        Dictionary of the ranges of the count features. Keys are the
        indexes of the continuous features, values are a tuple of minimum and
        maximum values.
    pop_size: int
        Number of individuals in the population.
    n_generatios: int
        Number of generations.
    n_elites:
        Number of best individuals to keep from one generation to the other.
    tournsize: int
        Number of individuals that participate on a tournament.
    n_hall_of_fame: int
        Number of individuals to keep in the hall of fame.
    n_runs: int
        Number of runs (evolitionary processes).
    count_features_as_categorical: bool
        Whether to consider count features as categorical features or not.
    weight_cat: float
        Weight for the similarity measure between categorical features of the
        synthetic observation and the original observation.
    weight_cont: float
        Weight for the similarity measure between continuous features of the
        synthetic observation and the original observation.
    weight_input_loss: float
        Weight for the part of the loss function related to the similarity
        between the features' values.
    weight_prediction_loss: float
        Weight for the part of the loss function related to the similarity
        between the predicted probabilities.
    early_stopping_threshold:
        Threshold to stop the evolutionary process. If all the individuals in
        the hall of fame have a fitness value greater than the threhsold, the
        evolutionary process is stopped.
    distance: str
        Distance metric to use for continuous variables. It can be 'cosine' or
        'ned' (normalized euclidean distance).
    """

    predict_proba_fn: Callable
    X_obs: np.ndarray
    threshold: float
    features_names: List[str]
    categorical_features_indexes: List[int] = field(default_factory=[])
    continuous_features_indexes: List[int] = field(default_factory=[])
    count_features_indexes: List[int] = field(default_factory=[])
    categorical_features_values: Dict[int, Tuple] = field(default_factory={})
    continuous_features_range: Dict[int, Tuple] = field(default_factory={})
    count_features_range: Dict[int, Tuple] = field(default_factory={})
    pop_size: int = 100
    n_generations: int = 50
    n_elites: int = 5
    tournsize: int = 10
    n_hall_of_fame: int = 10
    n_runs: int = 5
    count_features_as_categorical: bool = True
    weight_cat: float = 0.5
    weight_cont: float = 0.5
    weight_input_loss: float = 0.5
    weight_prediction_loss: float = 0.5
    early_stopping_threshold: float = 0.99
    distance: str = "cosine"

    def _check_inputs(self):
        """
        Checks the validity of the input arguments.
        """
        indexes_input = (
            self.continuous_features_indexes
            or self.categorical_features_indexes
            or self.count_features_indexes
        )

        assert (
            indexes_input
        ), "Please, provide one of categorical or continuous features names."

        if self.categorical_features_indexes:
            assert (
                self.categorical_features_values
            ), "Please, provide the possible values for categorical features."

        if self.count_features_indexes:
            assert (
                self.count_features_range
            ), "Please, provide the range of values for count features."

        if self.continuous_features_indexes:
            assert (
                self.continuous_features_range
            ), "Please, provide the range of values for continuous features."

    def __post_init__(self):
        """
        Set arguments and initialize the toolbox.
        """
        self._check_inputs()

        self.predicted_class = np.argmax(
            self.predict_proba_fn(self.X_obs.reshape(1, -1))[0, :]
        )

        self.cat_columns = (
            self.categorical_features_indexes + self.count_features_indexes
            if self.count_features_as_categorical
            else self.categorical_features_indexes
        )

        self.cont_columns = (
            self.continuous_features_indexes + self.count_features_indexes
            if not self.count_features_as_categorical
            else self.continuous_features_indexes
        )

        if not self.cat_columns and self.weight_cat != 0.0:
            warnings.warn(
                "Weight for categorical columns set to 0 because no categorical columns were provided."
            )
            self.weight_cat = 0.0
            self.weight_cont = 1.0

        if not self.cont_columns and self.weight_cont != 0.0:
            warnings.warn(
                "Weight for continuous columns set to 0 because no categorical columns were provided."
            )
            self.weight_cont = 0.0
            self.weight_cat = 1.0

        self.toolbox = self._init_toolbox()

    def _init_population(self) -> list:
        """
        Population initialization.
        """
        pop = []
        for _ in range(self.pop_size):
            genes = []
            for i in range(len(self.X_obs)):
                if i in self.categorical_features_indexes:
                    genes.append(
                        CounterfactualGene(
                            [np.random.choice(self.categorical_features_values[i])]
                        )
                    )
                elif i in self.count_features_indexes:
                    genes.append(
                        CounterfactualGene(
                            [
                                np.random.randint(
                                    self.count_features_range[i][0],
                                    self.count_features_range[i][1] + 1,
                                )
                            ]
                        )
                    )
                elif i in self.continuous_features_indexes:
                    genes.append(
                        CounterfactualGene(
                            [
                                np.random.uniform(
                                    self.continuous_features_range[i][0],
                                    self.continuous_features_range[i][1] + 1,
                                )
                            ]
                        )
                    )
                else:
                    # fix feature value (constant gene)
                    genes.append(CounterfactualGene([self.X_obs[i]]))
            ind = Chromosome.from_genes(genes)
            ind.fitness = FitnessMax()
            pop.append(ind)

        return pop

    def _init_toolbox(self) -> Toolbox:
        """
        Toolbox initialization.
        """
        toolbox = Toolbox()

        toolbox.register("compile", counterfactual_compile)

        toolbox.register("select", selTournament, tournsize=self.tournsize)

        if self.continuous_features_indexes:
            toolbox.register("mut_continuous", self._mutate_continuous, pb=0.1)
        if self.categorical_features_indexes:
            toolbox.register("mut_categorical", self._mutate_categorical, pb=0.1)
        if self.count_features_indexes:
            toolbox.register("mut_count", self._mutate_count, pb=0.1)

        toolbox.register("cx_gene", self._crossover_gene, pb=0.1)

        return toolbox

    def _mutate_categorical(self, ind: Chromosome) -> Chromosome:
        """
        Mutation operator for categorical features.
        """
        col = np.random.choice(self.categorical_features_indexes)

        ind[col] = CounterfactualGene(
            [np.random.choice(self.categorical_features_values[col])]
        )

        return ind

    def _mutate_continuous(self, ind: Chromosome) -> Chromosome:
        """
        Mutation operator for continuous fetures.
        """
        col = np.random.choice(self.continuous_features_indexes)

        ind[col] = CounterfactualGene(
            [
                np.random.uniform(
                    self.continuous_features_range[col][0],
                    self.continuous_features_range[col][1] + 1.0,
                )
            ]
        )

        return ind

    def _mutate_count(self, ind: Chromosome) -> Chromosome:
        """
        Mutation operator for count features.
        """
        col = np.random.choice(self.count_features_indexes)
        ind[col] = CounterfactualGene(
            [
                np.random.randint(
                    self.count_features_range[col][0],
                    self.count_features_range[col][1] + 1,
                )
            ]
        )
        return ind

    def _crossover_gene(self, ind1: Chromosome, ind2: Chromosome) -> Tuple[Chromosome]:
        """
        Crossover operator
        """
        pos = np.random.choice(range(len(ind1)))

        ind1_copy = copy(ind1)
        ind2_copy = copy(ind2)

        ind1_copy[pos] = ind2[pos]
        ind2_copy[pos] = ind1[pos]

        return ind1_copy, ind2_copy

    def feature_importance(
        self, best_individuals: pd.DataFrame, X: pd.DataFrame, n_bins: int = 20
    ) -> pd.DataFrame:
        """
        Computes the feature importance value, defined as:

            feat_importance_i = sum(g_i)/c_i,

        where

            i: feature i
            g_i = 1 if X_orig_i != X_synthetic_i else 0
            c_i: cardinality of feature i

        The continuous features are discretized using n_bins.

        Parameters
        ----------
        best_individuals: pd.DataFrame
            Pandas DataFrame with the best individuals (as produced by the
            'explained' method).
        X: np.ndarray
            Features matrix containing all the available observations.
        n_bins: int
            Number of bins for discretizing the continuous features.

        Returns
        -------
        feat_imp: pd.DataFrame
            Features importances.
        """
        X_transformed = X.copy()

        transformer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")

        X_transformed.iloc[
            :, self.continuous_features_indexes
        ] = transformer.fit_transform(
            X_transformed.iloc[:, self.continuous_features_indexes]
        )

        Xs = best_individuals.drop("fitness", axis=1).copy().values
        Xs[:, self.continuous_features_indexes] = transformer.transform(
            Xs[:, self.continuous_features_indexes]
        )

        X_orig = self.X_obs.copy()
        X_orig[self.continuous_features_indexes] = transformer.transform(
            X_orig[self.continuous_features_indexes].reshape(1, -1)
        )[0, :]

        g = (Xs != X_orig).sum(axis=0) / X_transformed.nunique().values

        feat_imp = pd.DataFrame(
            {"column": self.features_names, "feature_importance": g}
        )

        return feat_imp.sort_values(by="feature_importance", ascending=False)

    def explain(self) -> pd.DataFrame:
        """
        Run the counterfactual explanation process

        Returns
        -------
        best_individuals: pd.DataFrame
            Pandas DataFrame with the best individuals (best synthetic
            observations) and their respective fitness value.
        """
        best_individuals = []

        for _ in tqdm(range(self.n_runs)):

            hof = HallOfFame(self.n_hall_of_fame)

            self.toolbox.population = self._init_population()

            population = gep_counterfactual(
                self.toolbox,
                self.predict_proba_fn,
                self.X_obs,
                self.distance,
                self.predicted_class,
                self.cat_columns,
                self.cont_columns,
                self.n_generations,
                self.n_elites,
                self.early_stopping_threshold,
                hof,
                self.weight_cat,
                self.weight_cont,
                self.weight_input_loss,
                self.weight_prediction_loss,
            )

            best_individuals.extend([ind for ind in hof])

        fitnesses = np.array([ind.fitness.values[0] for ind in best_individuals])
        best_individuals = np.array(
            [self.toolbox.compile(ind) for ind in best_individuals]
        )

        best_individuals = pd.DataFrame(
            np.concatenate([best_individuals, fitnesses.reshape(-1, 1)], axis=1),
            columns=self.features_names + ["fitness"],
        )

        best_individuals.sort_values(by="fitness", ascending=False, inplace=True)

        print("Original Observation:")
        print(pd.Series(self.X_obs, index=self.features_names))
        print("Best synthetic observation:")
        print(best_individuals.iloc[0])
        pred = self.predict_proba_fn(
            best_individuals.drop("fitness", axis=1).iloc[0].values.reshape(1, -1)
        )[0, :]
        print(f"Model prediction on synthetic observation: {pred}")

        return best_individuals
