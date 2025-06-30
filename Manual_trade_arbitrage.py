
from typing import List, Tuple
from math import log, exp

rates = [
    [1, 1.45, 0.52, 0.72],      # SNOW
    [0.7, 1, 0.31, 0.48],       # PIZ
    [1.95, 3.1, 1, 1.49],       # SI
    [1.34, 1.98, 0.64, 1]       # SS
]

currencies = ('SNOW', 'PIZ', 'SI', 'SS')


def negate_logarithm_convertor(graph: List[List[float]]) -> List[List[float]]:
    return [[-log(edge) for edge in row] for row in graph]


def find_arbitrage_from_SS(currencies: Tuple[str], rates_matrix: List[List[float]], steps: int = 5):
    graph = negate_logarithm_convertor(rates_matrix)
    n = len(graph)
    start = currencies.index("SS")

    # dp[k][i] = min cost to reach currency i in k steps from SS
    dp = [[float('inf')] * n for _ in range(steps + 1)]
    path = [[[] for _ in range(n)] for _ in range(steps + 1)]

    dp[0][start] = 0
    path[0][start] = [start]

    # Fill DP table
    for k in range(1, steps + 1):
        for u in range(n):  # from
            if dp[k-1][u] < float('inf'):
                for v in range(n):  # to
                    cost = dp[k-1][u] + graph[u][v]
                    if cost < dp[k][v]:
                        dp[k][v] = cost
                        path[k][v] = path[k-1][u] + [v]

    # Now check if returning to 'SS' after `steps` gives arbitrage
    if dp[steps][start] < 0:
        cycle = path[steps][start]
        print("Arbitrage Opportunity (5 operations, starting and ending at SS):")
        print(" --> ".join([currencies[i] for i in cycle]))
        print(f"Profit multiplier: {exp(-dp[steps][start]):.4f}")
    else:
        print("No arbitrage opportunity of exactly 5 operations starting and ending at SS found.")


if __name__ == "__main__":
    find_arbitrage_from_SS(currencies, rates, steps=5)
