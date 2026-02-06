import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


class EESD1:
    """
    Extended Exponential Smoothing (d=1) with drift.
    X_T = (I + λ ∇'(I - u(u'u)^-1 u')∇)^(-1) X
    b   = (u'u)^-1 u' ∇ X_T
    """

    def __init__(self, lam: float, use_log: bool = True):
        self.lam = float(lam)
        self.use_log = use_log
        self.result_ = None

    def fit(self, price: pd.Series):
        s = price.dropna().astype(float)
        if len(s) < 5:
            raise ValueError("Need at least 5 observations.")
        X = np.log(s.values) if self.use_log else s.values
        T = len(X)

        # 1차 차분 ∇
        e = np.ones(T)
        D = sparse.diags([e, -e], [0, -1], shape=(T - 1, T), format="csc")

        # u: 차분 공간 상수벡터
        u = np.ones((T - 1, 1))
        utu = float(u.T @ u)  # = T-1
        I_diff = sparse.identity(T - 1, format="csc")
        P = I_diff - (sparse.csc_matrix(u) @ sparse.csc_matrix(u.T)) / utu

        # 선형시스템
        I_T = sparse.identity(T, format="csc")
        A = I_T + self.lam * (D.T @ (P @ D))
        XT = spsolve(A, X)

        dXT = (D @ XT).reshape(-1, 1)
        b = float((u.T @ dXT) / utu)
        resid = X - XT

        self.result_ = {
            "X": pd.Series(X, index=s.index, name="X"),
            "XT": pd.Series(XT, index=s.index, name="XT"),
            "b": b,
            "dXT": pd.Series(np.r_[np.nan, np.diff(XT)], index=s.index, name="dXT"),
            "resid": pd.Series(resid, index=s.index, name="X_minus_XT"),
        }
        return self.result_

    @staticmethod
    def trend_signals(result: dict, slope_window: int = 1):
        dXT = result["dXT"].copy()
        if slope_window > 1:
            dXT = dXT.rolling(slope_window).mean()
        direction = np.sign(dXT)   # +1 상승, -1 하락
        strength = dXT.abs()       # 절댓값이 클수록 강한 추세
        return pd.DataFrame(
            {"dXT": dXT, "direction": direction, "strength": strength, "resid": result["resid"]}
        )


class EESD2:
    """
    HP-style trend filter (d=2).
    X_T = (I + lambda * D2' D2)^(-1) X
    """

    def __init__(self, lam: float, use_log: bool = True):
        self.lam = float(lam)
        self.use_log = use_log
        self.result_ = None

    def fit(self, price: pd.Series):
        s = price.dropna().astype(float)
        if len(s) < 5:
            raise ValueError("Need at least 5 observations.")
        X = np.log(s.values) if self.use_log else s.values
        T = len(X)

        # 2차 차분 D2
        e = np.ones(T)
        # Row i represents X_{i+2} - 2*X_{i+1} + X_i
        D2 = sparse.diags([e, -2 * e, e], [0, 1, 2], shape=(T - 2, T), format="csc")

        I_T = sparse.identity(T, format="csc")
        A = I_T + self.lam * (D2.T @ D2)
        XT = spsolve(A, X)

        resid = X - XT
        self.result_ = {
            "X": pd.Series(X, index=s.index, name="X"),
            "XT": pd.Series(XT, index=s.index, name="XT"),
            "dXT": pd.Series(np.r_[np.nan, np.diff(XT)], index=s.index, name="dXT"),
            "resid": pd.Series(resid, index=s.index, name="X_minus_XT"),
        }
        return self.result_

def lambda_sweep_plateau(price: pd.Series,
                         lambdas,
                         eps_quantile: float = 0.2,
                         min_run: int = 2,
                         use_log: bool = True):
    """
    여러 λ로 d=1 추세 계산 후, 연속 λ 사이 추세 변화량 D_i로 플래토 탐지.
    D_i = ||XT_{i+1} - XT_i||_2 / (||XT_i||_2 + 1e-12)
    """
    lambdas = sorted(list(lambdas))
    XT_dict, XT_list = {}, []

    for lam in lambdas:
        out = EESD2(lam=lam, use_log=use_log).fit(price)
        XT = out["XT"].values
        XT_dict[lam] = out["XT"]
        XT_list.append(XT)

    D_vals = []
    for i in range(len(lambdas) - 1):
        a, b = XT_list[i], XT_list[i + 1]
        num = np.linalg.norm(b - a)
        den = np.linalg.norm(a) + 1e-12
        D_vals.append(num / den)
    D = pd.Series(D_vals, index=lambdas[:-1], name="D_change")

    eps = float(D.quantile(eps_quantile))
    is_small = (D <= eps).values

    best = (None, None, 0)
    i = 0
    while i < len(is_small):
        if not is_small[i]:
            i += 1
            continue
        j = i
        while j < len(is_small) and is_small[j]:
            j += 1
        length = j - i
        if length >= min_run and length > best[2]:
            best = (i, j, length)
        i = j

    plateau_lams = []
    if best[0] is not None:
        start, end = best[0], best[1]
        plateau_lams = lambdas[start: end + 1]  # end+1 포함

    return XT_dict, D, plateau_lams
