# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MIPipeline(BaseEstimator, ClassifierMixin):

    def __init__(self, members=None, feature_names=None):
        self.members = members if members is not None else []
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.classes_ = np.array([0, 1])

    # ------------------------------------------------------------------
    def _check_input(self, X):
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            raise ValueError(f"输入缺少特征列：{missing}")
        return X

    def _member_positive_proba(self, X, member):
        x_enc = member["preprocessor"].transform(X[self.feature_names])
        return member["model"].predict_proba(x_enc)[:, 1]

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        X = self._check_input(X)
        if len(self.members) == 0:
            raise ValueError("members 为空，请先在训练环境构建并保存。")
        proba_each = [self._member_positive_proba(X, m) for m in self.members]
        p1 = np.mean(np.vstack(proba_each), axis=0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # 供网页端 SHAP 使用的辅助接口
    # ------------------------------------------------------------------
    @property
    def n_imputations(self):
        return len(self.members)

    def transform_member(self, X, idx=0):
        X = self._check_input(X)
        return self.members[idx]["preprocessor"].transform(X[self.feature_names])

    def model_member(self, idx=0):
        return self.members[idx]["model"]

    def preprocessor_member(self, idx=0):
        return self.members[idx]["preprocessor"]
