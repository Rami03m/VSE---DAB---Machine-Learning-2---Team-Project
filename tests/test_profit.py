from src.profit import BusinessParams, delta_profit

def test_delta_profit_positive_when_accuracy_improves():
    p = BusinessParams(m_operations=0.03, k=0.3, alpha=1000, epsilon=3.0, b=12.0)
    dp = delta_profit(0.6, 0.7, p)
    assert dp > 0.0