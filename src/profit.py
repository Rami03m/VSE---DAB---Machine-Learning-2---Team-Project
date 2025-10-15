from dataclasses import dataclass
import numpy as np
from typing import Optional, Callable

@dataclass
class BusinessParams:
    m_operations: float  # fixed cost margin
    k: float             # risk coeff for m_risk(A) = k*(1 - A)
    alpha: float         # demand scale
    epsilon: float       # price elasticity (>1)
    b: float             # avg bet size (currency units)

def total_margin(m_profit: float, A: float, p: BusinessParams) -> float:
    return m_profit + p.m_operations + p.k * (1.0 - A)

def demand(m: float, p: BusinessParams) -> float:
    return p.alpha * (m ** (-p.epsilon))

def profit(m_profit: float, A: float, p: BusinessParams) -> float:
    m = total_margin(m_profit, A, p)
    return demand(m, p) * m_profit * p.b

def argmax_profit_numeric(A: float, p: BusinessParams,
                          m_profit_max: float = 0.5,  # reasonable cap
                          grid_points: int = 2000) -> tuple[float, float]:
    grid = np.linspace(1e-6, m_profit_max, grid_points)
    vals = np.array([profit(x, A, p) for x in grid])
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])

def delta_profit(A0: float, A1: float, p: BusinessParams) -> float:
    m0, pi0 = argmax_profit_numeric(A0, p)
    m1, pi1 = argmax_profit_numeric(A1, p)
    return pi1 - pi0