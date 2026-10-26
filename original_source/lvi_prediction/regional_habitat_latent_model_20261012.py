"""PCA/PLS compression followed by regularized LR, composed into exact raw-feature logits."""
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression


class LatentLogisticModel:
    def __init__(self,method,n_components,C):
        self.method,self.n_components,self.C = method,n_components,C

    def fit(self,x,y):
        n = min(self.n_components,x.shape[1],len(x)-1)
        if self.method=="PCA":
            self.reducer_ = PCA(n_components=n,svd_solver="full").fit(x)
            z = self.reducer_.transform(x)
            projection,center = self.reducer_.components_.T,self.reducer_.mean_
        else:
            self.reducer_ = PLSRegression(n_components=n,scale=False,max_iter=1000).fit(x,y)
            z = self.reducer_.transform(x)
            projection = self.reducer_.x_rotations_/self.reducer_._x_std[:,None]
            center = self.reducer_._x_mean
        self.latent_mean_ = z.mean(0)
        self.latent_scale_ = np.where(z.std(0)>1e-12,z.std(0),1.)
        self.classifier_ = LogisticRegression(C=self.C,penalty="l2",solver="liblinear",max_iter=3000,random_state=64212).fit(
            (z-self.latent_mean_)/self.latent_scale_,y)
        w = self.classifier_.coef_[0]/self.latent_scale_
        self.coef_ = (projection@w)[None,:]
        self.intercept_ = np.asarray([self.classifier_.intercept_[0]-center@self.coef_[0]-self.latent_mean_@w])
        self.classes_ = np.array([0,1])
        direct = self.predict_proba(x)
        reference = self.classifier_.predict_proba((z-self.latent_mean_)/self.latent_scale_)
        np.testing.assert_allclose(direct,reference,atol=1e-10,rtol=0)
        return self

    def predict_proba(self,x):
        p = expit(x@self.coef_[0]+self.intercept_[0])
        return np.column_stack([1-p,p])
