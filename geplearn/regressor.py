from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
import pandas as pd
import numpy as np
from typing import Union
from numba import cuda
from .base import GEPBase


class GEPRegressor(GEPBase, RegressorMixin):
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Makes a prediction on the input X. If linear scaling is True, the
        prediction is scaled using the learned parameters.

        Parameters
        ----------
        X: Union[pd.DataFrame, np.ndarray]
            Features matrix

        Returns
        -------
        y: np.ndarray
            Model prediction.
        """
        check_is_fitted(self, "is_fitted_")

        X = check_array(X)

        X_ = (
            [
                cuda.to_device(np.ascontiguousarray(X[:, i]))
                for i in range(self.n_features)
            ]
            if self.target_device == "cuda"
            else X
            if self.n_features > 32
            else [X[:, i] for i in range(self.n_features)]
        )

        if self.target_device == "cuda":

            yp = self.prediction_function(*X_)

            yp = yp.copy_to_host()

        elif self.n_features > 32:

            yp = self.prediction_function(X_)

        else:

            yp = self.prediction_function(*X_)

        if self.linear_scaling:
            yp = self.best_individual.a * yp + self.best_individual.b

        return yp
