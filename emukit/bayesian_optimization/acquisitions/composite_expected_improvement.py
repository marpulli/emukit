from typing import Callable, Union

import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel, IDifferentiable


class CompositeExpectedImprovement(Acquisition):
    """
    This is the EI-CF from: Bayesian Optimization of Composite Functions (Raul Astudillo, Peter I. Frazier)

    It is used when optimizing the functions of the form g(h(x)) where g is a fast to evaluate deterministic
    transformation of some unknown function h(x).
    """

    def __init__(self, model: Union[IModel, IDifferentiable], deterministic_transformation_fcn: Callable,
                 n_monte_carlo_samples: int = 20):
        """
        :param model: The model of h(x), the unknown, slow to evaluate function
        :param deterministic_transformation_fcn: A python function for the fast to evaluate transformation g(.),
            This function should take a numpy array of dimension (n_points x n_model_outputs) and return a numpy array
            of shape (n_points, )
        :param n_monte_carlo_samples: Number of monte carlo sample to use when estimating the acquisition function value
        """
        self.model = model
        self.deterministic_transformation_fcn = deterministic_transformation_fcn
        self.standard_normal_samples = np.random.randn(n_monte_carlo_samples)

    def evaluate(self, x):
        """
        Evaluate the acquisition function

        :param x:
        :return:
        """
        mean, var = self.model.predict(x)
        y_min = np.min(self.deterministic_transformation_fcn(self.model.Y))
        std = np.atleast_3d(np.sqrt(var))

        predictive_distribution_samples = mean[:, :, None] + std * self.standard_normal_samples[None, None, :]

        # Loop over samples to evaluate g(predictive_distribution_samples)
        h_samples = np.zeros((x.shape[0], self.standard_normal_samples.shape[0]))
        for i in range(self.standard_normal_samples.shape[0]):
            h_samples[:, i] = self.deterministic_transformation_fcn(predictive_distribution_samples[:, :, i]).flatten()

        return np.mean(np.maximum(y_min - h_samples, 0), axis=1)

    def evaluate_with_gradients(self, x: np.ndarray):
        """

        :param x:
        :return:
        """
        mean, var = self.model.predict(x)
        dmean_dx, dvar_dx = self.model.get_prediction_gradients(x)

        # Compute gradients of standard deviation wrt x
        dstd_dx = 0.5 * dvar_dx / np.clip(np.sqrt(var), 1e-8, np.inf)
        y_min = np.min(self.model.Y)
        predictive_distribution_samples = mean + np.sqrt(var) * self.standard_normal_samples[None, :]
        is_below_y_min = predictive_distribution_samples < y_min

        grad = np.zeros(x.shape)
        grad[is_below_y_min, :] = dmean_dx + dstd_dx * self.standard_normal_samples
        return np.mean(grad, axis=2)

    @property
    def has_gradients(self):
        """
        :return: Whether the acquisition function can provide gradients
        """
        return False
        #return isinstance(self.model, IDifferentiable)
