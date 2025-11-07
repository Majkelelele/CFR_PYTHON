import numpy as np
import random

# mapping rock == 0, paper == 1 and scissors == 2

# custom value when any player chooses scissors we get value multiplied by 2

VALUE_MATRIX = np.array([
    [0,  1, -2],
    [-1, 0,  2],
    [2, -2,  0],
], dtype=np.float64)


# standard RPS
# VALUE_MATRIX = np.array([
#     [0,  1, -1],
#     [-1, 0,  1],
#     [1, -1,  0],
# ], dtype=np.float64)


class Player:
    def __init__(self):
        self.actions_count = 3
        self.regret_sum = [0] * self.actions_count
        self.strategy_sum = np.zeros(self.actions_count, dtype=np.float64)
        self.tol = 1e-4

        
        
    def get_strategy(self):
        regrets = np.asarray(self.regret_sum, dtype=np.float64)
        pos = np.clip(regrets, 0.0, None)          
        s = pos.sum()

        if s > 0:
            curr_strategy = pos / s
        else:
            curr_strategy = np.full(self.actions_count, 1.0 / self.actions_count, dtype=np.float64)
        self.strategy_sum += curr_strategy

        return curr_strategy
        
    def pick_action(self, strategy) -> int:
        cum_sum = np.cumsum(strategy)
        
        assert abs(cum_sum[-1] - 1) < self.tol, f"{cum_sum=}"
        x = random.random()

        for i in range(self.actions_count):
            if cum_sum[i] >= x:
                return i
        raise ValueError("Wrong action picked")
        
            
    def average_strategy(self):
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.actions_count, 1.0 / self.actions_count, dtype=np.float64)
        
    
def train(iterations=10000):
    p1 = Player()
    p2 = Player()
    
    # pseudo check if initial strategy matters for convergence
    # p1.regret_sum[0] = 1
    # p2.regret_sum[2] = 1

    
    for j in range(iterations):
        p1_strategy = p1.get_strategy()
        p2_strategy = p2.get_strategy()
        a1 = p1.pick_action(p1_strategy)
        a2 = p2.pick_action(p2_strategy)

        
        u1 = VALUE_MATRIX[a2, :]                 
        p1.regret_sum += u1 - u1[a1]

        u2 = -VALUE_MATRIX[:, a1]                
        p2.regret_sum += u2 - u2[a2]
            
            
    return p1.average_strategy(), p2.average_strategy()







if __name__ == "__main__":
    p1_strategy, p2_strategy = train(iterations=10**5)
    print(f"{p1_strategy=}")

    print(f"{p2_strategy=}")
    