from sklearn.feature_selection import VarianceThreshold


class VarianceThresholdWrapper(VarianceThreshold):
    def fit_transform(self, X, y=None, **kwargs):
        super().fit_transform(X, y, **kwargs)
        return X[X.columns[self.get_support(indices=True)]]

    def transform(self, X):
        super().transform(X)
        return X[X.columns[self.get_support(indices=True)]]
