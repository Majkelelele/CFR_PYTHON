import numpy as np
import random

PASS, BET = 0,1
SEED = 42
JACK, QUEEN, KING = (0,1,2)
CARDS = [JACK, QUEEN, KING]
ACTIONS = [PASS, BET]

class Kuhn_I_node():
    def __init__(self, player, card, previous_actions, actions):
        self.strategy = np.full(len(actions), 1 / len(actions))
        self.regret_sum = np.zeros(len(actions))
        self.strategy_sum = np.zeros(len(actions))
        self.player = player
        self.actions: list[int] = actions
        self.previous_actions = previous_actions
        self.card = card
    
    def __str__(self):
        return f"{self.previous_actions}_Player{self.player}_Card{self.card}"
   
    def do_print(self):
        print(f"{self}, strategy average = {self.get_average_strategy()}")

    def update_strategy(self):
        self.strategy = np.clip(self.regret_sum, 0, None)
        s = np.sum(self.strategy)
        if s > 0:
            self.strategy /= s
        else:
            self.strategy = np.full(len(self.actions), 1 / len(self.actions))
        
    def get_average_strategy(self):
        s = np.sum(self.strategy_sum)
        res = self.strategy_sum / s if s > 0 else np.array([1/len(self.actions)] * len(self.actions))
        return np.round(res,3)
        

class Trainer():
    def __init__(self):
        self.nodes = {}
        self.cards = np.array(CARDS)
        self.make_infostates()
        


    def make_infostates(self):
        # PLAYER 1 first decision - 3 possible Info states
        for card in CARDS:
            node = Kuhn_I_node(0, card, [], [PASS, BET])
            self.nodes[str(node)] = node
        # PLAYER 2 - 6 possible states - 3 cards and bet/pass from player 1
        # bet
        for card in CARDS:
            node = Kuhn_I_node(1, card, [BET], [PASS, BET])
            self.nodes[str(node)] = node
        # pass
        for card in CARDS:
            node = Kuhn_I_node(1, card, [PASS], [PASS, BET])
            self.nodes[str(node)] = node
            
        # back to 1st player after first player passed and 2nd bet
        for card in CARDS:
            node = Kuhn_I_node(0, card, [PASS, BET], [PASS, BET])
            self.nodes[str(node)] = node
        for i in self.nodes:
            print(i)
    
    def terminal_val(self, hist):
        # P1 pass, P2 pass -> showdown, pot = 2
        if hist == "00":
            return 1 if self.cards[0] > self.cards[1] else -1

        # P1 bet, P2 fold -> P1 wins 1
        if hist == "10":
            return 1

        # P1 bet, P2 bet -> showdown, pot = 4
        if hist == "11":
            return 2 if self.cards[0] > self.cards[1] else -2

        # P1 pass, P2 bet, P1 fold -> P2 wins 1
        if hist == "010":
            return -1

        # P1 pass, P2 bet, P1 call -> showdown, pot = 4
        if hist == "011":
            return 2 if self.cards[0] > self.cards[1] else -2

        raise ValueError(f"Unknown terminal history: {hist}")

    
    def get_infostate(self, hist):
        player = self.get_player(hist)
        prev_actions = [int(i) for i in hist]
        s = f"{prev_actions}_Player{player}_Card{self.cards[player]}"
        return self.nodes.get(s, None)
        
    
    def get_player(self,hist):
        return len(hist) % 2
    
    def shuffle(self):
        for c1 in range(len(self.cards) - 1, -1, -1):
            c2 = random.randint(0, c1)
            self.cards[c1], self.cards[c2] = self.cards[c2], self.cards[c1]

    def cfr(self, hist, p_1, p_2):
        info_s: Kuhn_I_node = self.get_infostate(hist)
        if not info_s:
            return self.terminal_val(hist)
    
        v_total = 0
        v_a = [0] * len(info_s.actions)
        for a in info_s.actions:
            if self.get_player(hist) == 0:
                v_a[a] = -self.cfr(hist + str(a), p_1 * info_s.strategy[a], p_2)
                p_minus_i = p_2
                p_i = p_1
            else:
                v_a[a] = -self.cfr(hist + str(a), p_1, p_2 * info_s.strategy[a])
                p_minus_i = p_1
                p_i = p_2
        
            v_total += v_a[a] * info_s.strategy[a]
        
        for a in info_s.actions:
            info_s.regret_sum[a] += p_minus_i * (v_a[a] - v_total) 
            info_s.strategy_sum[a] += p_i * info_s.strategy[a]
        
        info_s.update_strategy()
        return v_total
    
    
    
    def train(self,iteration=10**5):
        for _ in range(iteration):
            self.shuffle()
            self.cfr("",1,1)
        
        
        for i in self.nodes.values():
            i.do_print()
                
            
            
        

if __name__ == "__main__":
    random.seed(SEED)
    trainer = Trainer()
    trainer.train()
    
    

