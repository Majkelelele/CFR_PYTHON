import numpy as np
import random


class Trainer:
    def __init__(self, iterations=10000):
        # mapping rock == 0, paper == 1 and scissors == 2
        self.actions_count = 3
        self.regret_sum = [0] * self.actions_count
        self.iterations = iterations
        self.other_strategy = np.array([1/4,1/4,1/2], dtype=np.float64)
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
        
    
    def train(self):
        for j in range(self.iterations):
            curr_strategy = self.get_strategy()
            a = self.pick_action(curr_strategy)
            a_other = self.pick_action(self.other_strategy)
            
            actions_utility = [0] * self.actions_count
            actions_utility[(a_other + 1) % self.actions_count] = 1
            actions_utility[(a_other + 2) % self.actions_count] = -1
            
            for i in range(self.actions_count):
                self.regret_sum[i] += actions_utility[i] - actions_utility[a]
                
            if j % (10**4) == 0:        
                print(f"{curr_strategy=}")







if __name__ == "__main__":
    trainer = Trainer(iterations=100000)
    trainer.train()
    print("Final average strategy (σ̄):", np.round(trainer.average_strategy(), 4))

    