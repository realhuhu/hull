from typing import Protocol
from itertools import product
from functools import partial
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import numpy as np
import pandas as pd

from .enhance import BasePrimitive, Dataset
from .metric import score
from .pipeline import Pipeline


class Model(Protocol):
    def fit(self, x_train, y_train):
        ...

    def predict(self, x_test) -> pd.DataFrame:
        ...


class WeightFunc(Protocol):
    def __call__[T](self, y: T) -> T:
        ...


class Optimizer:
    valid_size = 126
    valid_num = 9

    def __init__(
            self,
            dataset: pd.DataFrame,
            primitives: list[BasePrimitive],
            drop_columns: list[str],
            cut_off=5400,
    ):
        self.primitives = primitives

        self.r = dataset[["risk_free_rate", "forward_returns"]]
        self.x, self.y = self.preprocess(dataset.drop(drop_columns, axis=1))

        self.x = self.x[-cut_off:]
        self.y = self.y[-cut_off:]
        self.r = self.r[-cut_off:]

        self.weight_func_list = []
        self.model_clz_list = []
        self.look_back_list = []

        self.pipeline = Pipeline([])

    def baseline(self):
        hold_res = []
        prev_res = []
        rand_res = []
        for i in range(self.valid_num):
            divider = len(self.x) - (i + 1) * self.valid_size
            r_test = self.r[divider:divider + self.valid_size]
            y_test = self.y[divider:divider + self.valid_size]

            hold_res.append(score(r_test, 1))
            prev_res.append(score(r_test, (y_test > 0).shift(1).astype("boolean").fillna(False)))
            rand_res.append(sum(score(r_test, np.random.rand(len(y_test)) > 0.5) for _ in range(100)) / 100)

        return {
            "hold": np.mean(hold_res),
            "prev": np.mean(prev_res),
            "rand": np.mean(rand_res),
        }

    def score(self):
        res = pd.DataFrame(columns=["model", "look_back", "weight_func", "score"])
        for m, l, f in tqdm(list(product(self.model_clz_list, self.look_back_list, self.weight_func_list))):
            res.loc[len(res)] = [
                m.func.__name__ if isinstance(m, partial) else m.__name__,
                l,
                f.__name__,
                self.score_one(m, l, f),
            ]

        return res.loc[:, res.nunique() != 1]  # 删除只有唯一值的列

    def preprocess(self, dataset: pd.DataFrame):
        e = Dataset(dataset.drop("target", axis=1))
        e.add_primitives(self.primitives)
        return e.build(), dataset["target"]

    def split(self, divider, look_back, pipeline):

        if look_back:
            x_train = self.x[divider - look_back:divider]
            y_train = self.y[divider - look_back:divider]
        else:
            x_train = self.x[:divider]
            y_train = self.y[:divider]

        x_test = self.x[divider:divider + self.valid_size]
        y_test = self.y[divider:divider + self.valid_size]
        r_test = self.r[divider:divider + self.valid_size]

        x_train, y_train = pipeline.fit_transform(x_train, y_train)
        x_test, y_test = pipeline.transform(x_test, y_test)

        return x_train, y_train, x_test, y_test, r_test

    def score_one(self, model_clz: type[Model], look_back: int, weight_func):
        scores = []
        for i in range(self.valid_num):
            divider = len(self.x) - (i + 1) * self.valid_size
            x_train, y_train, x_test, y_test, r_test = self.split(divider, look_back, self.pipeline)
            model = model_clz()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            scores.append(score(r_test, weight_func(y_pred)))

        return np.mean(scores)

    def report_one(self, model_clz: type[Model], look_back: int, weight_func, pipeline):
        results = pd.DataFrame(columns=["sharpe", "hold_sharpe", "prev_sharpe", "rand_sharpe"])

        for i in range(self.valid_num):
            divider = len(self.x) - (i + 1) * self.valid_size
            x_train, y_train, x_test, y_test, r_test = self.split(divider, look_back, pipeline)
            model = model_clz()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            results.loc[f"{divider}:{divider + self.valid_size}"] = {
                "sharpe": score(r_test, weight_func(y_pred)),
                "hold_sharpe": score(r_test, 1),
                "prev_sharpe": score(r_test, (y_test > 0).shift(1).astype("boolean").fillna(False)),
                "rand_sharpe": sum(score(r_test, np.random.rand(x_test.shape[0]) > 0.5) for _ in range(100)) / 100,
            }

        results.loc["avg"] = results.mean()
        return results
