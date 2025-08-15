from typing import Iterator

import numpy


class DimReducer:

    def __init__(self, dim=2, way="PCA", **kwargs):
        self.dim = dim
        if way.upper() == "PCA":
            from sklearn.decomposition import PCA
            self.model = PCA(n_components=self.dim, **kwargs)
        elif way.upper() == "TSNE":
            from sklearn.manifold import TSNE
            self.model = TSNE(n_components=self.dim, **kwargs)
        elif way.upper() == "ICA":
            from sklearn.decomposition import FastICA
            self.model = FastICA(n_components=self.dim, whiten="unit-variance", **kwargs)
        else:
            raise ValueError(f"way {way} is not supported")

    def fit_transform(self, *x: numpy.ndarray):
        """
        Args:
            *x: (n_samples, n_features)
        Return:
            (n_samples, dim)
        """
        y = self.vstack_transform_split(x, self.model.fit_transform)
        return y[0] if len(x) == 1 else y

    def transform(self, *x: numpy.ndarray):
        """
        Args:
            *x: (n_samples, n_features)
        Return:
            (n_samples, dim)
        """
        y = self.vstack_transform_split(x, self.model.transform)
        return y[0] if len(y) == 1 else y

    @staticmethod
    def vstack_transform_split(x: numpy.ndarray | Iterator[numpy.ndarray], transform_func) \
            -> numpy.ndarray | list[numpy.ndarray]:
        if isinstance(x, numpy.ndarray):
            return transform_func(x)
        x_lens = [len(xi) for xi in x]
        stacked_x = numpy.vstack(x)
        y = transform_func(stacked_x)
        y = numpy.split(y, numpy.cumsum(x_lens)[:-1])
        return y

    def reset(self):
        self.model = self.model.__class__(n_components=self.dim)
