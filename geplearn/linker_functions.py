from typing import List


def prepare_inputs(input_names: List[str], genes: List[str]):

    args = ", ".join(input_names)

    genes_joined = "+ ".join(genes)

    types = ", ".join(["float64" for _ in range(len(input_names))])

    return args, genes_joined, types


def sum_linker_template(
    input_names: List[str],
    input_names_reversed: List[str],
    genes: List[str],
    target: str,
):

    args, genes_joined, types = prepare_inputs(input_names, genes)

    func_str = (
        f"@vectorize('float64({types})', target='{target}')\n"
        f"def predict({args}):\n"
        f"    return {genes_joined}"
    )

    return func_str


def avg_linker_template(
    input_names: List[str],
    input_names_reversed: List[str],
    genes: List[str],
    target: str,
):

    args, genes_joined, types = prepare_inputs(input_names, genes)

    n_genes_inv = 1.0 / len(genes)

    func_str = (
        f"@vectorize('float64({types})', target='{target}')\n"
        f"def predict({args}):\n"
        f"    return {n_genes_inv}*({genes_joined})"
    )

    return func_str


def sum_linker_template_numba(
    input_names: List[str],
    input_names_reversed: List[str],
    genes: List[str],
    target: str,
):
    _, genes_joined, _ = prepare_inputs(input_names, genes)

    input_names_modified = {}
    for input_name in input_names:
        _, j = input_name.split("_")
        input_names_modified[input_name] = f"X[i, {j}]"

    genes_joined_modified = genes_joined
    for k in input_names_reversed:
        genes_joined_modified = genes_joined_modified.replace(
            k, input_names_modified[k]
        )

    func_str = (
        f"@njit(parallel={True if target == 'parallel' else False}, nogil=True)\n"
        f"def predict(X):\n"
        f"    yp = np.zeros(shape=X.shape[0])\n"
        f"    for i in prange(X.shape[0]):\n"
        f"        yp[i] = {genes_joined_modified}\n"
        f"    return yp"
    )

    return func_str


def avg_linker_template_numba(
    input_names: List[str],
    input_names_reversed: List[str],
    genes: List[str],
    target: str,
):
    _, genes_joined, _ = prepare_inputs(input_names, genes)

    input_names_modified = {}
    for input_name in input_names:
        _, j = input_name.split("_")
        input_names_modified[input_name] = f"X[i, {j}]"

    genes_joined_modified = genes_joined
    for k in input_names_reversed:
        genes_joined_modified = genes_joined_modified.replace(
            k, input_names_modified[k]
        )

    n_inputs_inv = 1.0 / len(input_names)

    func_str = (
        f"@njit(parallel={True if target == 'parallel' else False}, nogil=True)\n"
        f"def predict(X):\n"
        f"    yp = np.zeros(shape=X.shape[0])\n"
        f"    for i in prange(X.shape[0]):\n"
        f"        yp[i] = {n_inputs_inv}*({genes_joined_modified})\n"
        f"    return yp"
    )

    return func_str


LINKER_FUNCTIONS = {
    "numba": {
        "sum_linker": sum_linker_template_numba,
        "avg_linker": avg_linker_template_numba,
    },
    "vectorized": {
        "sum_linker": sum_linker_template,
        "avg_linker": avg_linker_template,
    },
}

