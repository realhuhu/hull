import pandas as pd
from abc import ABC, abstractmethod
import numpy as np


class BasePrimitive(ABC):
    def __init__(self, columns: list[str] = None):
        self.pos_columns = [column for column in columns or [] if not column.startswith("~")]
        self.neg_columns = [column[1:] for column in columns or [] if column.startswith("~")]

    @abstractmethod
    def build_col(self, col_name: str, col: pd.Series) -> dict[str, pd.Series]:
        raise NotImplementedError()

    @abstractmethod
    def valid_col(self, col: pd.Series) -> bool:
        raise NotImplementedError()

    def valid_name(self, col_name: str) -> bool:
        if not self.pos_columns and not self.neg_columns:
            return True

        for neg_pattern in self.neg_columns:
            if neg_pattern.endswith('*'):
                prefix = neg_pattern[:-1]  # 去掉末尾的*
                if col_name.startswith(prefix) and len(col_name) > len(prefix):
                    return False
            else:
                if col_name == neg_pattern:
                    return False

        if not self.pos_columns:
            return True

        for pos_pattern in self.pos_columns:
            if pos_pattern.endswith('*'):
                prefix = pos_pattern[:-1]
                if col_name.startswith(prefix):
                    return True
            else:
                if col_name == pos_pattern:
                    return True

        # 如果没有匹配任何包含规则，则不包含该列
        return False

    def build(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)
        col_names = []
        for col_name in columns:
            col = df[col_name]

            if not self.valid_col(col):
                continue

            if not self.valid_name(col_name):
                continue

            col_names.append(col_name)
            res = res.join(pd.DataFrame(self.build_col(col_name, col), index=df.index))

        print(f"{self}处理的列: {col_names}")
        return res

    def __repr__(self):
        return self.__class__.__name__

    __str__ = __repr__


class BaseRolling(BasePrimitive):
    @abstractmethod
    def build_col(self, col_name: str, col: pd.Series):
        raise NotImplementedError()

    @abstractmethod
    def valid_col(self, col: pd.Series):
        raise NotImplementedError()

    def __init__(self, windows: list[int], columns: list[str] = None):
        super().__init__(columns)
        self.windows = windows


class RollingMin(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        return {
            f"{col_name}_Min_R{window}": col.rolling(window).min()
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class RollingMax(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        return {
            f"{col_name}_Max_R{window}": col.rolling(window).max()
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class RollingMean(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        return {
            f"{col_name}_Mean_R{window}": col.rolling(window).mean()
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class RollingStd(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        return {
            f"{col_name}_Std_R{window}": col.rolling(window).std()
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class RollingCountTrue(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        col = col.fillna(False)
        return {
            f"{col_name}_CtTrue_R{window}": col.rolling(window).sum()
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        if col.dtype.kind in "b":
            return True

        if isinstance(col.dtype, pd.CategoricalDtype) and col.dtype.categories.dtype == bool:
            return True

        return False


class RollingCountAboveMean(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        col_clean = col.dropna()

        return {
            f"{col_name}_CtAMean_R{window}": col_clean.rolling(window).apply(
                lambda x: (x > x.mean()).sum(), raw=True
            )
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class RollingTrend(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        def rolling_trend(data):
            if len(data) < 2:  # 窗口太小无法计算趋势
                return np.nan

            x = np.arange(len(data))  # 创建自变量 [0, 1, 2, ...]
            return np.polyfit(x, data, 1)[0]

        return {
            f"{col_name}_Trend_R{window}": col.rolling(window).apply(rolling_trend, raw=True)
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class RollingMaxConsecutiveTrue(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        col = col.fillna(False)

        def max_consecutive_true(x):
            if len(x) == 0:
                return np.nan

            arr = np.array(x, dtype=bool)
            changes = np.diff(np.concatenate(([False], arr, [False])).astype(int))
            run_starts = np.where(changes == 1)[0]
            run_ends = np.where(changes == -1)[0]
            run_lengths = run_ends - run_starts
            return np.max(run_lengths) if len(run_lengths) > 0 else 0

        return {
            f"{col_name}_MaxConsTrue_R{window}": col.rolling(window).apply(
                max_consecutive_true, raw=True
            )
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        if not col.dtype.kind in "b" and not (
                isinstance(col.dtype, pd.CategoricalDtype)
                and col.dtype.categories.dtype == bool
        ):
            return False

        if np.all(col.astype(bool)):
            return False

        if not np.any(col.astype(bool)):
            return False

        return True


class RollingMaxConsecutivePositives(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        col_clean = col.dropna()

        def max_consecutive_positives(x):
            if len(x) == 0:
                return np.nan

            arr = np.array(x)
            changes = np.diff(np.concatenate(([False], arr > 0, [False])).astype(int))
            run_starts = np.where(changes == 1)[0]
            run_ends = np.where(changes == -1)[0]
            run_lengths = run_ends - run_starts
            return np.max(run_lengths) if len(run_lengths) > 0 else 0

        return {
            f"{col_name}_MaxConsPos_R{window}": col_clean.rolling(window).apply(
                max_consecutive_positives, raw=True
            )
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        if col.dtype.kind not in "iuf":
            return False

        if np.all(col > 0):
            return False

        if not np.any(col > 0):
            return False

        return True


class RollingNumSinceLastTrue(BaseRolling):
    def build_col(self, col_name: str, col: pd.Series):
        col = col.fillna(False)

        def num_since_last_true(x):
            if len(x) == 0:
                return np.nan

            arr = np.array(x, dtype=bool)
            true_indices = np.where(arr)[0]

            if len(true_indices) == 0:
                return len(x)

            return len(x) - 1 - true_indices[-1]

        return {
            f"{col_name}_NslTrue_R{window}": col.rolling(window).apply(
                num_since_last_true, raw=True
            )
            for window in self.windows
        }

    def valid_col(self, col: pd.Series) -> bool:
        if col.dtype.kind in "b":
            return True

        if isinstance(col.dtype, pd.CategoricalDtype) and col.dtype.categories.dtype == bool:
            return True

        return False


class BaseEwm(BasePrimitive):
    @abstractmethod
    def build_col(self, col_name: str, col: pd.Series):
        raise NotImplementedError()

    @abstractmethod
    def valid_col(self, col: pd.Series) -> bool:
        raise NotImplementedError()

    def __init__(self, spans: list[int], columns: list[str] = None):
        super().__init__(columns)
        self.spans = spans


class EwmMean(BaseEwm):
    def build_col(self, col_name: str, col: pd.Series):
        return {
            f"{col_name}_Mean_E{span}": col.ewm(span=span).mean()
            for span in self.spans
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class EwmStd(BaseEwm):
    def build_col(self, col_name: str, col: pd.Series):
        return {
            f"{col_name}_Std_E{span}": col.ewm(span=span).std()
            for span in self.spans
        }

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class Lag(BasePrimitive):
    def __init__(self, period: int = 1, columns: list[str] = None):
        super().__init__(columns)
        self.period = period

    def build_col(self, col_name: str, col: pd.Series) -> dict[str, pd.Series]:
        return {f"{col_name}_Lag_P{self.period}": col.shift(self.period)}

    def valid_col(self, col: pd.Series) -> bool:
        return True


class Diff(BasePrimitive):
    def __init__(self, period: int = 1, columns: list[str] = None):
        super().__init__(columns)
        self.period = period

    def build_col(self, col_name: str, col: pd.Series):
        return {f"{col_name}_Diff_P{self.period}": col.diff(self.period)}

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class PctChange(BasePrimitive):
    def __init__(self, period: int = 1, columns: list[str] = None):
        super().__init__(columns)
        self.period = period

    def build_col(self, col_name: str, col: pd.Series):
        return {f"{col_name}_PctCh_P{self.period}": col.pct_change(self.period).replace([np.inf, -np.inf], 0)}

    def valid_col(self, col: pd.Series) -> bool:
        return col.dtype.kind in "iuf"


class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns
        self.primitives: list[BasePrimitive] = []

    def add_primitives(self, primitives: list[BasePrimitive]):
        self.primitives.extend(primitives)
        return self

    def build(self) -> pd.DataFrame:
        res = self.df.copy()
        for primitive in self.primitives:
            res = res.join(primitive.build(res, self.columns))

        return res


__all__ = [
    "BasePrimitive",
    "BaseRolling",
    "BaseEwm",
    "Dataset",
    "RollingMin",
    "RollingMax",
    "RollingMean",
    "RollingStd",
    "RollingCountTrue",
    "RollingCountAboveMean",
    "RollingTrend",
    "RollingMaxConsecutiveTrue",
    "RollingMaxConsecutivePositives",
    "RollingNumSinceLastTrue",
    "EwmMean",
    "EwmStd",
    "Lag",
    "Diff",
    "PctChange",
]
