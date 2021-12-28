from typing import Dict, List
from sympy.core.symbol import Symbol
from tqdm import tqdm
from sympy import lambdify, posify
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .base import GEPBase
from IPython.display import display


plt.style.use("seaborn-darkgrid")


def summarize_model(X: pd.DataFrame, gep_reg: GEPBase, cat_features: List[str]) -> None:
    """
    Prints a summary of the gep model.

    Parameters
    ----------
    X: pd.DataFrame
        DataFrame containing the all the available features.

    gep_reg: GEPBase
        GEP model with a symbolic function.

    cat_features: List[str]
        Categorical features names.

    Returns
    -------
    None
    """
    symbolic_function = gep_reg.symbolic_function

    features_symbols = list(symbolic_function.free_symbols)

    features_str = {symbol: str(symbol) for symbol in features_symbols}

    features_names = {
        symbol: X.columns[int(features_str[symbol].split("_")[1])]
        for symbol in features_symbols
    }

    print("-----Model Summary-----\n")
    print("Symbolic Function:")
    try:
        display(symbolic_function)
    except:
        print(symbolic_function)
    print()
    print("-" * 10)
    print(
        f"\nNumber of independent variables used by the gep model: {len(features_symbols)} of {X.shape[1]}\n"
    )
    print("-" * 10)
    features_cat = [
        symbol for symbol in features_symbols if features_names[symbol] in cat_features
    ]

    features_cont = list(filter(lambda x: x not in features_cat, features_symbols))

    if features_cat:
        print("\nCategorical features utilized by the gep model:\n")
        print({features_str[symbol]: features_names[symbol] for symbol in features_cat})

    if features_cont:
        print("\nContinuous features utilized by the gep model:\n")
        print(
            {features_str[symbol]: features_names[symbol] for symbol in features_cont}
        )


def compute_normalized_sensitivities(X: pd.DataFrame, gep_reg: GEPBase) -> pd.DataFrame:
    """
    Computes the normalized sensitivites, defined as the x0 times gradient of
    the funtion at x0, divided by the value of the function at x0.

    Parameters
    ----------
    X: pd.DataFrame
        Features DataFrame.
    gep_reg: GEPBase
        GEP model.

    Returns
    -------
    normalized_sensitivities: pd.DataFrame
        DataFrame containing the normalized sensitivities.
    """
    symbolic_function = gep_reg.symbolic_function

    features_symbols = list(symbolic_function.free_symbols)

    features_str = {symbol: str(symbol) for symbol in features_symbols}

    features_names = {
        symbol: X.columns[int(features_str[symbol].split("_")[1])]
        for symbol in features_symbols
    }

    features_str_inv = {v: k for k, v in features_str.items()}

    raw_sensitivities = {}

    values_dict = {
        features_str[k]: X.loc[:, features_names[k]] for k in features_symbols
    }

    for symbol in tqdm(features_symbols):

        partial_derivative = symbolic_function.diff(symbol)

        f = np.vectorize(lambdify(features_symbols, partial_derivative))

        raw_sensitivities[features_names[symbol]] = f(**values_dict)

    raw_sensitivities_df = pd.DataFrame(raw_sensitivities)

    y_predicted = gep_reg.predict(X)

    normalized_sensitivities = (
        X.loc[:, list(features_names.values())] * raw_sensitivities_df
    ).divide(y_predicted, axis=0)

    return normalized_sensitivities


def boxplot_normalized_sensitivities(normalized_sensitivities: pd.DataFrame) -> None:
    """
    Makes a boxplot of the normalized sensitivities.

    Parameters
    ----------
    normalized_sensitivities: pd.DataFrame
        Normalized sensitivities, obtained by using the function
        "compute_normalized_sensitivities".

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        x="variable",
        y="value",
        data=normalized_sensitivities.stack()
        .reset_index(level=1)
        .rename({"level_1": "variable", 0: "value"}, axis=1),
        showfliers=False,
        ax=ax,
    )

    ax.tick_params(axis="x", labelrotation=90)

    fig.tight_layout()


def plot_normalized_sensitivities_by_obs(
    X: pd.DataFrame, gep_reg: GEPBase, normalized_sensitivities: pd.DataFrame = None
) -> None:

    features_symbols = gep_reg.symbolic_function.free_symbols

    features_str = {symbol: str(symbol) for symbol in features_symbols}

    features_names = {
        symbol: X.columns[int(features_str[symbol].split("_")[1])]
        for symbol in features_symbols
    }

    if normalized_sensitivities is None:

        normalized_sensitivities = compute_normalized_sensitivities(X, gep_reg)

    symbols_range = list(range(len(features_symbols)))

    for ob in range(len(X)):

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.barh(symbols_range, normalized_sensitivities.iloc[ob].values, align="center")

        ax.set_yticks(symbols_range)

        ax.set_yticklabels(list(features_names.values()))

        ax.invert_yaxis()

        ax.set_xlabel("Normalized Sensitivity")

        ax.set_title(f"Sensitivity for observation {ob}")

        textstr = "\n".join(
            ["Features values\n"]
            + [
                f"{k}: {v:.2f}"
                for k, v in X.iloc[ob][list(features_names.values())].to_dict().items()
            ]
        )
        props = dict(boxstyle="square", facecolor="tab:blue", alpha=0.1, pad=1)

        ax.text(
            1.2,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

        fig.tight_layout()

        fig.show()


def averaged_effect(
    X: pd.DataFrame,
    gep_reg: GEPBase,
    feature: str,
    cat_features: List[str] = [],
    cat_feature_map: Dict[str, float] = {},
    rotation_x_tick_labels: float = 90,
) -> Symbol:
    """
    Plots the output of the model against a feature, when all the other
    features take the average/most frequent value.

    Parameters
    ----------
    X: pd.DataFrame
        Features Matrix.
    gep_reg: GEPBase
        GEP model.
    feature: str
        Name of the feature to inspect.
    cat_features: List[str]
        List of categorical features.
    cat_feature_map: Dict[str, float]
        Dictionary with the mapping from categories names to their
        encoded values.
    rotation_x_tick_labels: float
        Rotation of the labels on the x-axis.

    Returns
    -------
    symbolic_function: Symbol
        Symbolic function after replacing all the other features by their
        average/most frequent value.
    """
    symbolic_function = gep_reg.symbolic_function

    features_symbols = symbolic_function.free_symbols

    features_str = {symbol: str(symbol) for symbol in features_symbols}

    features_names = {
        symbol: X.columns[int(features_str[symbol].split("_")[1])]
        for symbol in features_symbols
    }

    features_names_inv = {v: k for k, v in features_names.items()}

    cat_features_map_inv = {v: k for k, v in cat_feature_map.items()}

    if feature in cat_features:
        X_vals = X.loc[:, feature].unique()
    else:
        min_val = X.loc[:, feature].min()
        max_val = X.loc[:, feature].max()

        X_vals = np.linspace(min_val, max_val, 200)

    for symbol in features_symbols:
        if features_names[symbol] != feature:

            subs_value = (
                X.loc[:, features_names[symbol]].mode()
                if features_names[symbol] in cat_features
                else X.loc[:, features_names[symbol]].mean()
            )

            symbolic_function = symbolic_function.subs(symbol, subs_value)

    try:
        display(symbolic_function)
    except:
        print(symbolic_function)

    symbolic_function = np.vectorize(
        lambdify(features_names_inv[feature], symbolic_function)
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    if feature in cat_features:

        ax.bar(range(len(X_vals)), symbolic_function(X_vals))
        ax.set_xticks(range(len(X_vals)))
        ax.set_xticklabels(
            [cat_features_map_inv[v] for v in X_vals],
            rotation=rotation_x_tick_labels,
            fontsize=12,
        )
    else:
        ax.plot(X_vals, symbolic_function(X_vals))

    ax.set_ylabel("Output", fontsize=14)

    ax.set_xlabel(feature, fontsize=14)

    if feature in cat_features and cat_feature_map:

        textstr = "\n".join(
            ["Categories Mapping\n"]
            + [f"{k}: {v:.2f}" for k, v in cat_feature_map.items()]
        )

        props = dict(boxstyle="square", facecolor="tab:blue", alpha=0.1, pad=1)

        ax.text(
            1.2,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

    fig.tight_layout()

    fig.show()

    return symbolic_function
