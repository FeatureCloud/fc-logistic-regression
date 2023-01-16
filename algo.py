import numpy as np
from scipy.special._ufuncs import expit

from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted


class Client(LogisticRegression):
    beta_global = None
    classes_ = None
    coef_ = None
    intercept_ = None
    n_features = None
    n_samples = None

    def set_coefs(self, coef):
        self.coef_ = np.asmatrix(coef[1:])
        self.intercept_ = np.ravel(coef[0])

    def init(self, X, y):
        column_one = np.ones((X.shape[0], 1)).astype(np.uint8)
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        X = np.concatenate((column_one, X), axis=1)
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        self.beta_global = np.zeros(X.shape[1])
        self.classes_ = np.unique(np.ravel(y))

        return X, y, self.beta_global

    def compute_derivatives(self, X, y, beta):
        self.beta_global = beta

        def gradient(X, y, beta):
            return X.T.dot((1 / (1 + 1 / np.exp(X.dot(beta))) - y))

        def hessian(X, beta):
            return X.T.dot((np.diag(np.ravel(np.exp(X.dot(beta)) / np.power(1 + np.exp(X.dot(beta)), 2))).dot(X)))

        grad = gradient(X, y, beta)
        hess = hessian(X, beta)

        return [grad, hess]

    def predict(self, X):
        scores = np.ravel(self.decision_function(X))
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)

        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        prob = np.ravel(self.decision_function(X))
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

    def predict_proba(self, X):
        check_is_fitted(self)

        ovr = (self.multi_class in ["ovr", "warn"] or
               (self.multi_class == 'auto' and (self.classes_.size <= 2 or
                                                self.solver == 'liblinear')))
        if ovr:
            return self._predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)


class Coordinator(Client):
    iteration_count = 0
    beta_global = None
    tol = 1e9

    def aggregate_beta(self, local_results):
        if "early_stop" in local_results:
            return self.beta_global, True
        self.iteration_count += 1

        gradients = [client[0] for client in local_results]
        hessians = [client[1] for client in local_results]

        gradient_global = gradients[0]
        for i in range(1, len(gradients)):
            gradient_global = gradient_global + gradients[i]

        hessian_global = hessians[0]
        for i in range(1, len(hessians)):
            hessian_global = hessian_global + hessians[i]

        updated_beta = self.beta_global - np.linalg.inv(hessian_global).dot(gradient_global)
        if np.isnan(updated_beta).any():
            print("Overflow error. Stopped early.")
            return self.beta_global, True
        else:
            self.beta_global = updated_beta
        if np.linalg.norm(gradient_global) > self.tol:
            finished = False
        else:
            finished = True
        return self.beta_global, finished
