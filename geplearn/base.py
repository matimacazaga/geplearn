from geppy.support.simplification import simplify
from geppy.tools.crossover import (
    crossover_gene,
    crossover_one_point,
    crossover_two_point,
)
from geppy.tools.mutation import (
    gene_transpose,
    invert,
    invert_dc,
    is_transpose,
    mutate_rnc_array_dc,
    mutate_uniform,
    mutate_uniform_dc,
    ris_transpose,
    transpose_dc,
)
from sklearn.base import BaseEstimator
from deap import tools as deap_tools
from sklearn.utils.validation import check_X_y
from dataclasses import dataclass
from typing import Callable, Optional, Union, Tuple
from numbers import Number
import random
from operator import add, sub, mul
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
from .evaluation_functions import rmse, correlation, sklearn_evaluation
from .linker_functions import LINKER_FUNCTIONS
from .config import SYMBOLIC_FUNCTION_MAP
from .gep_core import (
    CustomGene,
    CustomGeneDc,
    IndividualMin,
    IndividualMax,
    custom_compile,
    gep_simple,
    warmUpInit,
)
import numpy as np
from numba import cuda
from pandas import DataFrame
from multiprocessing import cpu_count
from geppy import PrimitiveSet, Toolbox, GeneDc, Gene


@dataclass
class GEPBase(BaseEstimator):
    """
    Base class for GEP models.

    Parameters
    ----------
    n_features: int
        Number of features in the problem.
    linker_function: Union[Callable, str]
        Function that links the genes in a Chromosome. It can be 'sum_linker',
        'avg_linker', or any Callable with a compatible structure (see
        linker_functions.py)
    functions_set: Tuple[Tuple[Callable, int]]
        Set of functions that can be part of an individual/gene. Each tuple
        must contain the function (Callable) and the arity (number of inputs).
    p_function: float
        Probability of assigning a function from the function set to a node.
    weights: np.ndarray
        Weights assigned to each function in the functions set for sampling. A
        greater weight indicates a greater probability. If None, all the
        functions have the same weight.
    fitness_func: Union[str, Callable]
        Evaluation/fitness function. It can be 'correlation', 'rmse', or any
        sklearn-like metric.
    pop_size: int
        Population size.
    warm_up: bool
        Whether to initialize the population using a 'warm up' step or not. If
        true, 10*pop_size indiviudals are generated, and the n fittest are
        selected for starting the evolutionary process.
    n_generations: int
        Number of generations.
    n_elites: int
        Number of best individuals to keep from one generation to the other.
    head_length: int
        Length of the genes' head.
    n_genes: int
        Number of genes per individual (chromosome).
    selection: str
        Selection process. It can be 'tournament' or 'roulette'. 'roulette'
        can only be used with maximization problems where the fitness value is
        non-negative.
    tournsize: int
        Number of individuals that participate of a tournament. Used if
        'selection' == 'tournament'.
    rnc_array_length: int
        Length of the random numerical constant (rnc) array (if linear scaling
        is True).
    rnc_rand_generator: Callable
        Function to generate the random numerical constants.
    rnc_range: Tuple[Number]
        Range of the rnc.
    parallel_ind_evaluation: int
        Number of individuals to evaluate in parallel. If it is equals to 0,
        the individuals are evaluated sequentially.
    n_runs: int
        Number of runs (evolutionary processes) to perform.
    parallel_runs: bool
        Whether to run in parallel the evolutionary processes or not. Only
        useful when n_runs>1.
    target_device: str
        Target device. It can be 'cpu' (single core), 'parallel' (multi-core),
        or 'gpu' (individuals are executed in the gpu).
    direction: str
        Whether to 'maximize' or 'minimize' the fitness function.
    linear_scaling: bool
        Whether to perform linear scaling or not.
    mut_* : float
        All the mut_* parameters indicate the probability of applying a
        mutation operator.
    cx_*: float
        All the cx_* parameters indicate the probability of applying a
        crossover operator.
    verbose: bool
        Whether to print statistics during fitting or not.
    is_fitted: bool
        Whether the model is already fitted or not.
    """

    n_features: int
    linker_function: Union[Callable, str] = "sum_linker"
    functions_set: Tuple[Tuple[Callable, int]] = ((add, 2), (sub, 2), (mul, 2))
    p_function: float = 0.5
    weights: Optional[np.ndarray] = None
    fitness_func: Union[str, Callable] = "correlation"
    pop_size: int = 30
    warm_up: bool = True
    n_generations: int = 100
    n_elites: int = 5
    head_length: int = 7
    n_genes: int = 3
    selection: str = "tournament"
    tournsize: int = 3
    rnc_array_length: int = 5
    rnc_rand_generator: Callable = random.randint
    rnc_range: Tuple[Number] = (-5.0, 5.0)
    parallel_ind_evaluation: int = 10
    n_runs: int = 2
    parallel_runs: bool = False
    target_device: str = "cpu"
    direction: str = "maximize"
    linear_scaling: bool = True
    mut_uniform: float = 1.0
    mut_uniform_ind_pb: float = 0.1
    mut_invert: float = 0.1
    mut_is_ts: float = 0.1
    mut_ris_ts: float = 0.1
    mut_gene_ts: float = 0.1
    cx_1p: float = 0.1
    cx_2p: float = 0.1
    cx_gene: float = 0.1
    mut_dc_ind_pb: float = 0.1
    mut_dc: float = 1.0
    mut_invert_dc: float = 0.1
    mut_transpose_dc: float = 0.1
    mut_rnc_array_dc_ind_pb: float = 0.5
    mut_rnc_array_dc: float = 1.0
    verbose: bool = True
    is_fitted_: bool = False

    def __post_init__(self,) -> None:
        """
        Creates usefull variables for individuals compiling, like target_device
        and input_names_reversed. It also creates the PrimitiveSet and the
        GEP Toolbox.
        """
        self.input_names = [f"X_{i}" for i in range(self.n_features)]

        self.input_names_reversed = sorted(
            self.input_names, key=lambda k: int(k.split("_")[1]), reverse=True
        )

        self.template_numba = self.n_features >= 32 and self.target_device != "cuda"

        if isinstance(self.linker_function, str):
            if self.template_numba:

                if self.linker_function not in LINKER_FUNCTIONS["numba"]:
                    raise NotImplementedError(
                        f"The linker function {self.linker_function} is not implemented."
                    )

                self.linker_function_ = LINKER_FUNCTIONS["numba"][self.linker_function]

            else:

                if self.linker_function not in LINKER_FUNCTIONS["vectorized"]:
                    raise NotImplementedError(
                        f"The linker function {self.linker_function} is not implemented."
                    )

                self.linker_function_ = LINKER_FUNCTIONS["vectorized"][
                    self.linker_function
                ]
        else:

            self.linker_function_ = self.linker_function

        if not self.weights:
            self.weights = np.array(
                [1.0 / len(self.functions_set) for _ in range(len(self.functions_set))]
            )

        pset = PrimitiveSet("Main", input_names=self.input_names)

        for f, arity in self.functions_set:
            pset.add_function(f, arity)

        if self.linear_scaling:
            pset.add_rnc_terminal()

        pset._globals["target"] = self.target_device

        pset._globals[
            "input_names_reversed"
        ] = (
            self.input_names_reversed
        )  #! Check if it is necessary to save this as a class argument

        self._create_toolbox(pset)

        self.stats = deap_tools.Statistics(key=lambda ind: ind.fitness.values[0])

        self.stats.register(
            "avg", lambda x: np.mean([xi for xi in x if not np.isnan(xi)])
        )

        self.stats.register(
            "std", lambda x: np.std([xi for xi in x if not np.isnan(xi)])
        )

        self.stats.register(
            "min", lambda x: np.min([xi for xi in x if not np.isnan(xi)])
        )

        self.stats.register(
            "max", lambda x: np.max([xi for xi in x if not np.isnan(xi)])
        )

    def __repr__(self,) -> str:

        if self.is_fitted_:
            return str(self.symbolic_function)
        return super().__repr__()

    def _create_toolbox(self, pset: PrimitiveSet) -> None:
        """
        Helper function to create the GEP toolbox. It registers the individual,
        population, genes and corresponding operators.

        Parameters
        ----------
        pset: PrimitiveSet
            Primitive set containing the functions set, inputs names and other
            information regarding the GEP algorithm.

        Returns
        -------
        None
        """
        self.toolbox = Toolbox()

        if self.linear_scaling:

            self.toolbox.register(
                "rnc_gen",
                self.rnc_rand_generator,
                a=self.rnc_range[0],
                b=self.rnc_range[1],
            )

            self.toolbox.register(
                "gene_gen",
                CustomGeneDc,
                pset=pset,
                head_length=self.head_length,
                rnc_gen=self.toolbox.rnc_gen,
                rnc_array_length=self.rnc_array_length,
                p_function=self.p_function,
                weights=self.weights,
            )

        else:

            self.toolbox.register(
                "gene_gen",
                Gene,
                pset=pset,
                head_length=self.head_length,
                p_function=self.p_function,
                weights=self.weights,
            )

        self.toolbox.register(
            "individual",
            IndividualMin if self.direction == "minimize" else IndividualMax,
            gene_gen=self.toolbox.gene_gen,
            n_genes=self.n_genes,
            linker=self.linker_function_,
        )

        if self.warm_up:
            self.toolbox.register(
                "population", warmUpInit, container=list, direction=self.direction
            )
        else:
            self.toolbox.register(
                "population", deap_tools.initRepeat, list, self.toolbox.individual,
            )

        self.toolbox.register("compile", custom_compile, pset=pset)

        if self.selection == "roulette":
            self.toolbox.register("select", deap_tools.selRoulette)
        elif self.selection == "tournament":
            self.toolbox.register(
                "select", deap_tools.selTournament, tournsize=self.tournsize
            )
        else:
            raise NotImplementedError(
                f"The selection method {self.selection} is not implemented. Please use 'tournament' or 'roulette'."
            )

        self.toolbox.register(
            "mut_uniform",
            mutate_uniform,
            pset=pset,
            ind_pb=self.mut_uniform_ind_pb,
            pb=self.mut_uniform,
        )

        self.toolbox.register("mut_invert", invert, pb=self.mut_invert)

        self.toolbox.register("mut_is_ts", is_transpose, pb=self.mut_is_ts)

        self.toolbox.register("mut_ris_ts", ris_transpose, pb=self.mut_ris_ts)

        self.toolbox.register("mut_gene_ts", gene_transpose, pb=self.mut_gene_ts)

        self.toolbox.register("cx_1p", crossover_one_point, pb=self.cx_1p)

        self.toolbox.register("cx_2p", crossover_two_point, pb=self.cx_2p)

        self.toolbox.register("cx_gene", crossover_gene, pb=self.cx_gene)

        if self.linear_scaling:

            self.toolbox.register(
                "mut_dc", mutate_uniform_dc, ind_pb=self.mut_dc_ind_pb, pb=self.mut_dc
            )

            self.toolbox.register("mut_invert_dc", invert_dc, pb=self.mut_invert_dc)

            self.toolbox.register(
                "mut_transpose_dc", transpose_dc, pb=self.mut_transpose_dc
            )

            self.toolbox.register(
                "mut_rnc_array_dc",
                mutate_rnc_array_dc,
                rnc_gen=self.toolbox.rnc_gen,
                ind_pb=self.mut_rnc_array_dc_ind_pb,
                pb=self.mut_rnc_array_dc,
            )

    def _register_evaluation_function(self, X_input: np.ndarray, y: np.ndarray) -> None:
        """
        Helper function to register the evaluation function in the toolbox.

        Parameters
        ----------
        X_input: np.ndarray
            Features matrix.
        y: np.ndarray
            Targets.
        """
        if isinstance(self.fitness_func, str):

            if self.fitness_func not in ["rmse", "correlation"]:

                raise NotImplementedError(
                    f"The fitness function {self.fitness_func} is not implemented. Please use 'correlation' or 'rmse', or pass a sklearn compatible function."
                )

            eval_fun = rmse if self.fitness_func == "rmse" else correlation

            self.toolbox.register(
                "evaluate",
                lambda individual: eval_fun(
                    individual, self.toolbox, X_input, y, self.linear_scaling
                ),
            )

        else:

            self.toolbox.register(
                "evaluate",
                lambda individual: sklearn_evaluation(
                    individual,
                    self.toolbox,
                    X_input,
                    y,
                    self.fitness_func,
                    self.linear_scaling,
                ),
            )

    def _fit_runs(self, parallel: bool = False) -> None:
        """
        Run the GEP evolutionary process n times and saves the best individual
        and the final populations.

        Parameters
        ----------
        parallel: bool
            Whether to run the process in parallel or not.

        Returns
        -------
        None
        """
        if parallel:
            pops = Parallel(n_jobs=min(self.n_runs, cpu_count()))(
                delayed(gep_simple)(
                    self.toolbox.population(toolbox=self.toolbox, n=self.pop_size)
                    if self.warm_up
                    else self.toolbox.population(n=self.pop_size),
                    self.toolbox,
                    self.n_generations,
                    self.n_elites,
                    self.stats,
                    None,
                    self.verbose,
                    self.parallel_ind_evaluation,
                )
                for _ in range(self.n_runs)
            )

        else:
            pops = [
                gep_simple(
                    self.toolbox.population(toolbox=self.toolbox, n=self.pop_size)
                    if self.warm_up
                    else self.toolbox.population(n=self.pop_size),
                    self.toolbox,
                    self.n_generations,
                    self.n_elites,
                    self.stats,
                    None,
                    self.verbose,
                    self.parallel_ind_evaluation,
                )
                for _ in tqdm(range(self.n_runs))
            ]

        individuals = list(itertools.chain.from_iterable(pops))

        f = min if self.direction == "minimize" else max

        self.best_individual = f(individuals, key=lambda ind: ind.fitness.values[0])

        self.population = sorted(
            individuals,
            key=lambda ind: ind.fitness.values[0],
            reverse=False if self.direction == "minimize" else True,
        )

    def _simple_fit(self):
        """
        Runs the evolutionary process once and saves the best individual and the
        final population.
        """
        hall_of_fame = deap_tools.HallOfFame(1)

        pop = gep_simple(
            self.toolbox.population(toolbox=self.toolbox, n=self.pop_size)
            if self.warm_up
            else self.toolbox.population(n=self.pop_size),
            self.toolbox,
            n_generations=self.n_generations,
            n_elites=self.n_elites,
            stats=self.stats,
            hall_of_fame=hall_of_fame,
            parallel_ind_evaluation=self.parallel_ind_evaluation,
            verbose=self.verbose,
        )

        self.best_individual = hall_of_fame[0]

        self.population = pop

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Run the GEP evolutionary process to find the fittest individual.

        Parameters
        ----------
        X: np.ndarray
            Features matrix.
        y: np.ndarray
            Target vector.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)

        X_input = (
            X
            if self.template_numba
            else [
                cuda.to_device(np.ascontiguousarray(X[:, i])) for i in range(X.shape[1])
            ]
            if self.target_device == "cuda"
            else [X[:, i] for i in range(X.shape[1])]
        )

        self._register_evaluation_function(X_input, y)

        if self.n_runs > 1:

            self._fit_runs(self.parallel_runs)

        else:

            self._simple_fit()

        self.prediction_function = self.toolbox.compile(self.best_individual)

        try:
            self.symbolic_function = simplify(
                self.best_individual, symbolic_function_map=SYMBOLIC_FUNCTION_MAP
            )

            if self.linear_scaling:

                self.symbolic_function = (
                    self.best_individual.a * self.symbolic_function
                    + self.best_individual.b
                )

        except Exception as e:

            print(e)

            self.symbolic_function = str(self.best_individual)

        self.is_fitted_ = True

        return self

    def predict(self, X: Union[np.ndarray, DataFrame]):
        """
        Make a prediction using the best individual.
        """
        raise NotImplementedError("Please, implement the predict method.")
