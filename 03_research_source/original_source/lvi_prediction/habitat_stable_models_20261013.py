"""Ridge bootstrap average with auditable linear member probabilities."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


class BootstrapRidge:
    def __init__(self, C, seed, n_members=15):
        self.C, self.seed, self.n_members = C, seed, n_members

    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        self.members_, self.draws_ = [], []
        for _ in range(self.n_members):
            ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
            m = LogisticRegression(C=self.C, penalty='l2', solver='liblinear', max_iter=3000, random_state=64212).fit(x[ix], y[ix])
            self.members_.append(m)
            self.draws_.append(ix)
        self.coefficients_ = np.vstack([m.coef_[0] for m in self.members_])
        self.intercepts_ = np.array([m.intercept_[0] for m in self.members_])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        p = expit(x@self.coefficients_.T+self.intercepts_).mean(1)
        return np.column_stack([1-p, p])
