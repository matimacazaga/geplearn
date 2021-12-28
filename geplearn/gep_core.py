from deap.base import Fitness
from geppy import Chromosome, PrimitiveSet, Gene, GeneDc, generate_dc
from typing import Any, Callable, List, Optional, Tuple
from geppy.tools.toolbox import Toolbox
from numba import vectorize, prange, njit
from geppy.tools.parser import _compile_gene
from geppy.tools._util import _choose_a_terminal
import numpy as np
from geppy.algorithms.basic import (
    _validate_basic_toolbox,
    _apply_crossover,
    _apply_modification,
)
from deap.tools import Logbook, selBest
from joblib import Parallel, delayed
from deap.tools import Statistics, HallOfFame
from .evaluation_functions import counterfactuals_evaluation
import random


class FitnessMin(Fitness):
    weights = (-1,)

    def __init__(self, values: Tuple[float] = ()):
        super().__init__(values)


class FitnessMax(Fitness):
    weights = (1,)

    def __init__(self, values: Tuple[float] = ()):
        super().__init__(values)


class IndividualMin(Chromosome):
    def __init__(self, gene_gen: Callable, n_genes: int, linker: Callable = None):
        super().__init__(gene_gen, n_genes, linker)
        self.fitness = FitnessMin()
        self.a = 0.0  # from linear scaling
        self.b = 0.0  # from linear scaling


class IndividualMax(Chromosome):
    def __init__(self, gene_gen: Callable, n_genes: int, linker: Callable = None):
        super().__init__(gene_gen, n_genes, linker)
        self.fitness = FitnessMax()
        self.a = 0.0  # from linear scaling
        self.b = 0.0  # from linear scaling


class CounterfactualGene(Gene):
    def __init__(self, genome):

        self._head_length = 1
        self._tail_length = 0
        list.__init__(self, genome)


def custom_generate_genome(
    pset: PrimitiveSet, head_length: int, p_function: float, weights: np.ndarray
) -> List[Any]:

    h = head_length
    functions = pset.functions
    terminals = pset.terminals

    n_max = max(p.arity for p in functions)  # max arity
    t = h * (n_max - 1) + 1
    expr = [None] * (h + t)
    # head part: initialized with both functions and terminals
    for i in range(h):
        if random.random() < p_function:
            expr[i] = np.random.choice(functions, p=weights / np.sum(weights))
        else:
            expr[i] = _choose_a_terminal(terminals)
    # tail part: only terminals are allowed
    for i in range(h, h + t):
        expr[i] = _choose_a_terminal(terminals)
    return expr


class CustomGene(Gene):
    def __init__(self, pset, head_length, p_function, weights):

        self._head_length = head_length
        genome = custom_generate_genome(pset, head_length, p_function, weights)
        list.__init__(self, genome)


class CustomGeneDc(GeneDc):
    def __init__(
        self, pset, head_length, rnc_gen, rnc_array_length, p_function, weights
    ):
        self._head_length = head_length
        genome = custom_generate_genome(pset, head_length, p_function, weights)
        list.__init__(self, genome)
        t = head_length * (pset.max_arity - 1) + 1
        d = t
        # complement it with a Dc domain
        self._rnc_gen = rnc_gen
        dc = generate_dc(rnc_array_length, dc_length=d)
        self.extend(dc)
        # generate the rnc array
        self._rnc_array = [self._rnc_gen() for _ in range(rnc_array_length)]


def warmUpInit(container, direction, toolbox, n):

    inds = [toolbox.individual() for _ in range(10 * n)]

    ind_fitness = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in inds)

    ind_fitness = sorted(
        ind_fitness,
        key=lambda ind: ind.fitness.values[0],
        reverse=False if direction == "minimize" else True,
    )

    return container(ind_fitness[:n])


def counterfactual_compile(ind: Chromosome) -> np.ndarray:
    """
    Compiles an individual into a feature vector for counterfactual
    explanations

    Parameters
    ----------
    ind: Chromosome
        Individual to compile.

    Returns
    -------
    np.ndarray
        Feature vector obtained from the individual.
    """
    fs = [gene for gene in ind]

    if len(fs) == 1:
        return np.array(fs)
    else:
        return np.array([f[0] for f in fs])


def custom_compile(individual: Chromosome, pset: PrimitiveSet) -> Callable:
    """
    Compiles an individual into a python function (numba).

    Parameters
    ----------
    individual: Chromosome
        Individual to compile.
    pset: PrimitiveSet
        Primitive set containing the necessary global variables. Must contain
        "input_names_reversed" and "target".

    Returns
    -------
    g["predict"]: Callable
        Compiled individual.
    """
    linker = individual.linker

    if linker is None:

        fs = [_compile_gene(gene, pset) for gene in individual]

        if len(fs) == 1:
            return fs[0]
        else:
            args = ", ".join(pset.input_names)
            code = f"lambda {args}: tuple((f({args}) for f in fs))"
            return eval(code, pset.globals, {"fs": fs})

    fun_str = linker(
        pset.input_names,
        pset.globals["input_names_reversed"],
        [str(g) for g in individual],
        pset.globals["target"],
    )

    g = {
        **pset.globals,
        **{"vectorize": vectorize, "njit": njit, "np": np, "prange": prange},
    }

    exec(fun_str, g)

    return g["predict"]


def gep_simple(
    population: List[Chromosome],
    toolbox: Toolbox,
    n_generations: int = 100,
    n_elites: int = 5,
    stats: Statistics = None,
    hall_of_fame: HallOfFame = None,
    verbose: bool = False,
    parallel_ind_evaluation: int = 10,
) -> List[Chromosome]:
    """
    Runs the GEP evolutionary process.

    Parameters
    ----------
    population: List[Chromosome]
        Initial population.
    toolbox: Toolbox
        GEP toolbox.
    n_generations: int
        Number of generations.
    n_elites: int
        Number of elite individuals (they are copied to the next generation).
    stats: Statistics
        Object to save process' statistics.
    hall_of_fame: HallOfFame
        Object to keep record of the best individual.
    verbose: bool
        Wheter to print statistics or not.
    parallel_ind_evaluation: int
        Number of individuals to evaluate in parallel.

    Returns
    -------
    population: List[Chromosome]
        Final population.
    """
    _validate_basic_toolbox(toolbox)

    logbook = Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    for gen in range(n_generations + 1):

        invalid_individuals = [ind for ind in population if not ind.fitness.valid]

        population = [ind for ind in population if ind.fitness.valid]

        if parallel_ind_evaluation > 1:
            evaluated_individuals = Parallel(n_jobs=parallel_ind_evaluation)(
                delayed(toolbox.evaluate)(ind) for ind in invalid_individuals
            )
        else:
            evaluated_individuals = [
                toolbox.evaluate(ind) for ind in invalid_individuals
            ]

        population.extend(evaluated_individuals)

        if hall_of_fame is not None:
            hall_of_fame.update(population)

        record = stats.compile(population) if stats else {}

        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)

        if verbose:
            print(logbook.stream)

        if gen == n_generations:
            break

        elites = selBest(population, k=n_elites)

        offspring = toolbox.select(population, len(population) - n_elites)

        offspring = [toolbox.clone(ind) for ind in offspring]

        for op in toolbox.pbs:
            if op.startswith("mut"):
                offspring = _apply_modification(
                    offspring, getattr(toolbox, op), toolbox.pbs[op]
                )

        for op in toolbox.pbs:
            if op.startswith("cx"):
                offspring = _apply_crossover(
                    offspring, getattr(toolbox, op), toolbox.pbs[op]
                )

        population = elites + offspring

    return population


def apply_modification_counterfactual(population, operator, pb):

    for i in range(len(population)):
        if random.random() < pb:
            population[i] = operator(population[i])
            del population[i].fitness.values
    return population


def gep_counterfactual(
    toolbox: Toolbox,
    predict_proba_fn: Callable,
    X_obs: np.ndarray,
    distance: str,
    predicted_class: int,
    cat_columns: Optional[List[int]],
    cont_columns: Optional[List[int]],
    n_generations: int,
    n_elites: int,
    early_stopping_threshold: float,
    hall_of_fame: HallOfFame,
    weight_cat: float = 0.5,
    weight_cont: float = 0.5,
    threshold: float = 0.5,
    weight_input_loss=0.5,
    weight_prediction_loss: float = 0.5,
):

    _validate_basic_toolbox(toolbox)

    population = toolbox.population

    for gen in range(n_generations + 1):
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]

        population = [ind for ind in population if ind.fitness.valid]

        scores = counterfactuals_evaluation(
            invalid_individuals,
            toolbox,
            predict_proba_fn,
            X_obs,
            distance,
            predicted_class,
            cat_columns,
            cont_columns,
            weight_cat,
            weight_cont,
            threshold,
            weight_input_loss,
            weight_prediction_loss,
        )

        for ind, score in zip(invalid_individuals, scores):
            ind.fitness.values = (score,)
            population.append(ind)

        hall_of_fame.update(population)

        if gen == n_generations:
            break

        if np.all(
            np.array([ind.fitness.values[0] for ind in hall_of_fame])
            >= early_stopping_threshold
        ):
            break

        # selection with elitism
        elites = selBest(population, k=n_elites)
        offspring = toolbox.select(population, len(population) - n_elites)

        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mutation
        for op in toolbox.pbs:
            if op.startswith("mut"):
                offspring = apply_modification_counterfactual(
                    offspring, getattr(toolbox, op), toolbox.pbs[op]
                )

        # crossover
        for op in toolbox.pbs:
            if op.startswith("cx"):
                offspring = _apply_crossover(
                    offspring, getattr(toolbox, op), toolbox.pbs[op]
                )

        # replace the current population with the offsprings
        population = elites + offspring

    return population
