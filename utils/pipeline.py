from abc import ABC, abstractmethod

import pandas as pd
from feature_engine import encoding


class BasePipe(ABC):
    @abstractmethod
    def fit_transform(self, x_train, y_train):
        ...

    @abstractmethod
    def transform(self, x_test, y_test):
        ...


class Pipe(BasePipe):
    def __init__(self, handler, force_df=False):
        self.handler = handler
        self.force_df = force_df

    def fit_transform(self, x_train, y_train):
        x_new = x_train.copy()
        x_trans = self.handler.fit_transform(x_new, y_train)

        if self.force_df:
            x_trans = pd.DataFrame(x_trans, columns=x_new.columns, index=x_new.index)

        return x_trans, y_train

    def transform(self, x_test, y_test):
        x_new = x_test.copy()
        x_trans = self.handler.transform(x_new)

        if self.force_df:
            x_trans = pd.DataFrame(x_trans, columns=x_new.columns, index=x_new.index)

        return x_trans, y_test


class Rename(BasePipe):
    def __init__(self):
        self.map = {}

    def fit_transform(self, x_train, y_train):
        x_new = x_train.copy()
        self.map = {v: f"v{k}" for k, v in enumerate(x_new.columns)}
        return x_new.rename(columns=self.map), y_train

    def transform(self, x_test, y_test):
        x_new = x_test.copy()
        x_new.columns = {v: f"v{k}" for k, v in enumerate(x_new.columns)}
        return x_new.rename(columns=self.map), y_test


class SelectColumns(BasePipe):
    def __init__(self, columns):
        self.columns = columns

    def fit_transform(self, x_train, y_train):
        return x_train[self.columns], y_train

    def transform(self, x_test, y_test):
        return x_test[self.columns], y_test


class MeanEncoder(BasePipe):
    def __init__(self):
        self.encoder = encoding.MeanEncoder(unseen="encode")
        self.has_category = False

    def fit_transform(self, x_train, y_train):
        self.has_category = x_train.select_dtypes("category").columns.any()

        if not self.has_category:
            return x_train, y_train

        x_new = x_train.copy()
        return self.encoder.fit_transform(x_new, y_train), y_train

    def transform(self, x_test, y_test):
        if not self.has_category:
            return x_test, y_test

        x_new = x_test.copy()
        return self.encoder.transform(x_new), y_test


class Pipeline(BasePipe):
    def __init__(self, pipes: list[BasePipe]):
        self.pipes = pipes

    def fit_transform(self, x_train, y_train):
        for pipe in self.pipes:
            x_train, y_train = pipe.fit_transform(x_train, y_train)

        return x_train, y_train

    def transform(self, x_test, y_test):
        for pipe in self.pipes:
            x_test, y_test = pipe.transform(x_test, y_test)

        return x_test, y_test
