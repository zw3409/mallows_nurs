import math
import random
from typing import List, Sequence

def hit_and_run_L2_step(
    sigma: Sequence[int],
    beta: float
) -> List[int]:
    n = len(sigma)
    b: List[float] = []
    for i in range(1, n + 1):
        log_u_i = math.log(random.random())
        b_i = sigma[i - 1] + log_u_i / (2 * beta * i)
        b.append(b_i)

    available_positions = list(range(n))
    tau: List[int] = [None] * n

    for symbol in range(1, n + 1):
        eligible = [pos for pos in available_positions if b[pos] <= symbol]
        if not eligible:
            raise RuntimeError(f"No eligible position found while placing symbol {symbol}")
        chosen_pos = random.choice(eligible)
        tau[chosen_pos] = symbol
        available_positions.remove(chosen_pos)

    return tau