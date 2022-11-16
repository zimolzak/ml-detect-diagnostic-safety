import numpy as np
import sklearn
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression

class UnbalancedLogisticEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=10,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.n_estimators = n_estimators
        self.estimators = []
        
        self.penalty=penalty
        self.dual=dual
        self.tol=tol
        self.C=C
        self.fit_intercept=fit_intercept
        self.intercept_scaling=intercept_scaling
        self.class_weight=class_weight
        self.random_state=random_state
        self.solver=solver
        self.max_iter=max_iter
        self.multi_class=multi_class
        self.verbose=verbose
        self.warm_start=warm_start
        self.n_jobs=n_jobs
        self.l1_ratio=l1_ratio

        
    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Multiclass not supported")
        
        split_X = []
        split_y = []
        for label in self.classes_:
            split_X.append(X[y == label].to_numpy())
            split_y.append(y[y == label])
        
        dominant_i = 1 if len(split_y[0]) < len(split_y[1]) else 0
        rare_i = 0 if dominant_i == 1 else 1
        sample_size = len(split_y[rare_i])
        
        self.estimators = []
        estimators = [LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            l1_ratio=self.l1_ratio
        ) for i in range(self.n_estimators)]
        
        # np.random.set_state(self.random_state)
        sy = np.array([split_y[dominant_i][0]] * sample_size)
        for est in estimators:
            sXi = np.random.randint(0, len(split_y[dominant_i]), size=sample_size)
            # print(X.shape, split_X[dominant_i].shape, split_y[dominant_i].shape, sXi.shape)
            sX = split_X[dominant_i][sXi,:]
            XX = np.concatenate((sX, split_X[rare_i]), axis=0)
            yy = np.concatenate((sy, split_y[rare_i]))
            self.estimators.append(est.fit(XX, yy))
        return self
        
    def predict(self, X):
        predicted_prob = self.predict_proba(X)
        return self.classes_.take(np.argmax(predicted_prob, axis=1), axis=0)
        
    def predict_proba(self, X):
        probs = [est.predict_proba(X) for est in self.estimators]
        return sum(probs) / self.n_estimators
        
    def predict_log_proba(self, X):
        probs = [est.predict_log_proba(X) for est in self.estimators]
        log_proba = probs[0]
        for i in range(1, len(probs)):
            log_proba = np.logaddexp(log_proba, probs[i])
        
        log_proba -= np.log(self.n_estimators)
        
        return log_proba
        