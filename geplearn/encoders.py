import numpy as np
import pandas as pd
from pandas.core.algorithms import isin, unique
from sklearn.base import TransformerMixin, BaseEstimator
from typing import List, Union
from dataclasses import dataclass
from sklearn.utils import check_random_state


@dataclass
class BaseTransformer(BaseEstimator, TransformerMixin):

    cols_to_transform: List[str] = None
    handle_missing_values: str = "value"
    handle_unknown_values: str = "value"

    def __post_init__(self):
        assert not isinstance(
            self.cols_to_transform, str
        ), "Please, provide columns to transform as a list."
        self.mapping = {}
        self.nan_value = "NaN"
        self.target_col = "target"

    @staticmethod
    def _is_category(dtype) -> bool:
        """
        Check if dtype is categorical.
        """
        return pd.api.types.is_categorical_dtype(dtype)

    def _get_obj_cols(self, df: pd.DataFrame) -> list:
        """
        Returns names of "object" or categorical columns in a DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            Input DataFrame

        Returns
        -------
        obj_cols: list
            List of columns names.
        """

        obj_cols = [
            df.columns.values[idx]
            for idx, dt in enumerate(df.dtypes)
            if dt in ["object", "int32", "int64"] or self._is_category(dt)
        ]

        return obj_cols

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper function to manage missing values in the DataFrame. If
        "handle_missing_values" == "error", it will raise a ValueError. If that
        is not the case, it will imput missing values with "NaN".
        """
        cols_nan_values = np.any(df.loc[:, self.cols_to_transform].isna(), axis=0)

        if np.any(cols_nan_values):

            nan_cols = cols_nan_values[cols_nan_values == True].index.tolist()

            if self.handle_missing_values == "error":

                raise ValueError(
                    f"DataFrame contains missing values in the following columns: {nan_cols}"
                )

            else:

                df.loc[:, nan_cols] = df.loc[:, nan_cols].fillna(self.nan_value)

        return df

    @staticmethod
    def _check_X_y(X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:

        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame."

        assert isinstance(
            y, (pd.Series, np.ndarray)
        ), "You must provide the target y as a pd.Series or np.ndarray."

        if isinstance(y, pd.Series):

            assert y.index.equals(
                X.index
            ), "X and y indeces are different. Try passing y as np.ndarray if the index is not relevant."

        elif isinstance(y, np.ndarray):
            len_X = len(X)
            len_y = len(y)
            assert (
                len_X == len_y
            ), f"X and y have different length ({len_X} vs {len_y})."

    def _prefit_step(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Concatenates the features and targets in one DataFrame and gets
        categorical columns if they were not provided by the user.

        Parameters
        ----------
        X: pd.DataFrame
            Input DataFrame with features to transform.
        y: pd.Series or np.ndarray
            Targets.

        Returns
        -------
        df: pd.Dataframe
            Concatenation of features and targets.
        """
        df = pd.concat([X, y], axis=1)

        df.columns = X.columns.tolist() + ["target"]

        if not self.cols_to_transform:
            self.cols_to_transform = self._get_obj_cols(
                df.drop(self.target_col, axis=1)
            )

        df = self._handle_missing_values(df)

        assert len(self.cols_to_transform), ValueError(
            "Could not recognize columns to transform. Please provide a list of columns to transform, or convert columns to categorical data types (int, str, etc.)"
        )

        return df

    def _pre_transform_steps(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if the transformer is fitted and the columns seen during training
        are in the input DataFrame, and handle missing values.

        Parameters
        ----------
        X: pd.DataFrame
            Input DataFrame. Must contain the columns to transform seen during
            fitting.

        Returns
        -------
        df: pd.DataFrame
            DataFrame after handling missing values.
        """
        assert self.is_fitted, "Transformer not fitted."

        if np.any([col not in X.columns for col in self.cols_to_transform]):
            raise ValueError(
                f"The DataFrame does not contain the same columns that were seen during fitting. Expected columns: {self.cols_to_transform}"
            )

        df = X.copy()

        df = self._handle_missing_values(df)

        if self.handle_unknown_values == "error":
            for col in self.cols_to_transform:
                unique_values = np.unique(df.loc[:, col])
                if np.any([val not in self.mapping[col] for val in unique_values]):
                    raise ValueError(f"The column '{col}' contains unknown values.")

        return df

    def transform(self, X: pd.DataFrame, y=None, **kwargs) -> pd.DataFrame:
        """
        Transformed the data using the mean values calculated during fitting.

        Parameters
        ----------
        df: pd.DataFrame
            Input DataFrame containing the columns to transform.

        Returns
        -------
        df_copy: pd.DataFrame
            DataFrame with the corresponding columns transformed.
        """
        df = self._pre_transform_steps(X)

        for col in self.cols_to_transform:

            df.loc[:, col] = df.loc[:, col].apply(
                lambda x: self.mapping[col].get(
                    x,
                    self._default_value
                    if self.handle_unknown_values == "value"
                    else np.nan,
                )
            )

        return df


@dataclass
class TargetEncoder(BaseTransformer):

    min_samples: int = 1
    smoothing_factor: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self._mean = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None, **kwargs):
        """
        Fits the transformer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the columns to transform.
        y: pd.Series or np.ndarray
            Array or Series containing the target values.

        Returns
        -------
        self
        """

        self._check_X_y(X, y)

        df = self._prefit_step(X, y)

        self._mean = self._default_value = df.loc[:, self.target_col].mean()

        for col in self.cols_to_transform:

            stats = (
                df.groupby(col)
                .agg({self.target_col: ["count", "mean"]})
                .droplevel(0, axis=1)
            )

            smooth = 1.0 / (
                1.0
                + np.exp(-(stats["count"] - self.min_samples) / self.smoothing_factor)
            )

            encoding = self._mean * (1.0 - smooth) + stats["mean"] * smooth
            # Ignore unique values (prevent overfitting)
            encoding[stats["count"] == 1] = 0

            encoding = encoding.to_dict()

            if self.handle_missing_values == "return_nan":
                encoding[self.nan_value] = np.nan

            self.mapping[col] = encoding

        self.is_fitted = True

        return self


@dataclass
class WoEEncoder(BaseTransformer):

    regularization: float = 1.0
    randomized: bool = False
    sigma: float = 0.05
    seed: int = 42

    def __post_init__(self):

        super().__post_init__()

        self._sum = None
        self._count = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None, **kwargs):
        """
        Fits the transformer to the data.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame containing the columns to transform.
        y: pd.Series or np.ndarray
            Array or Series containing the target values.

        Returns
        -------
        self
        """

        self._check_X_y(X, y)

        unique_classes = np.unique(y)

        assert len(unique_classes) == 2, "Multiclass problems are not supported."

        assert np.all(
            unique_classes == np.array([0, 1])
        ), "Please, provide binary target as 0 or 1."

        df = self._prefit_step(X, y)

        self._positive_class_sum = y.sum()

        self._obs_count = len(y)

        self._default_value = 0.0

        for col in self.cols_to_transform:

            stats = (
                df.groupby(col)
                .agg({self.target_col: ["count", "sum"]})
                .droplevel(0, axis=1)
            )

            numerator = ((stats["count"] - stats["sum"]) + 0.5) / (
                self._obs_count - self._positive_class_sum
            )

            denominator = (stats["sum"] + 0.5) / self._positive_class_sum

            encoding = np.log(numerator / denominator)

            # Ignore unique values (prevent overfitting)
            encoding[stats["count"] == 1] = 0

            if self.randomized:
                random_generator = check_random_state(self.seed)
                encoding += random_generator.normal(0, self.sigma, size=len(encoding))

            encoding = encoding.to_dict()

            if self.handle_missing_values == "return_nan":
                encoding[self.nan_value] = np.nan

            self.mapping[col] = encoding

        self.is_fitted = True

        return self
