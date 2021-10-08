import numpy as np
import random
import math
def mean_reward(history):
    return history[:,2] / np.maximum(1, history[:,1])

def ucb1(history, sigma = 0.5):
    """ ucb1 strategy for agent """
    """
    input individual player's history of price arms: each price's number of plays and total reward received
    history[i,1] : number of plays of price option i
    history[i,2] : total reward received from choosing price option i
    compute scores for each price option according to history
    output price selection with maximum score
    """
    
    def upper_bound(hist, t, sigma):
        return np.sqrt(2 * sigma * math.log(t) / hist[:,1])

    # pick random arm with no history
    not_pulled_arms = np.nonzero(history[:,1] == 0)
    if len(not_pulled_arms[0]) > 0:
 
        return [random.choice(not_pulled_arms[0])]

    # number of arm pulls
    t = np.sum(history[:,1]) + 1
    
    mean_scores = mean_reward(history)
    
    scores = mean_scores / 1 + upper_bound(history, t, sigma)
    best_score = np.max(scores)
    best_arms = np.nonzero(scores >= (best_score - 0.00001))
    return [random.choice(best_arms[0])]

def ucb2(history, alpha = 0.5):
    
    
    def tau(alpha,r):
        return np.ceil(np.power(1 + alpha,r))
    
    def upper_bound2(t, alpha, r):
        return np.sqrt((1 + alpha) * np.log(math.e * t / tau(alpha,r)) / (2 * tau(alpha,r)))

    # pick random arm with no history
    not_pulled_arms = np.nonzero(history[:,1] == 0)
    if len(not_pulled_arms[0]) > 0:
        
        return [random.choice(not_pulled_arms[0]), -1]
    
    if history[1, 6] != 0 and history[1, 6] != -1:
        return [history[0, 6], history[1, 6] - 1]
    
    # number of arm pulls
    t = np.sum(history[:,1]) + 1
    
    mean_scores = mean_reward(history)
    r = history[:,5]
    scores = mean_scores / 1 + upper_bound2(t, alpha, r)
    best_score = np.max(scores)
    best_arms = np.nonzero(scores >= (best_score - 0.00001))
    
    return [random.choice(best_arms[0]), tau(alpha,r[int(random.choice(best_arms[0]))]+1) - tau(alpha,r[int(random.choice(best_arms[0]))]) - 1]

def epsilon_greedy(history, epsilon = 0.025):
    """ epsilon greedy strategy for agent """

    # pick random arm with no history
    not_pulled_arms = np.nonzero(history[:,1] == 0)
    if len(not_pulled_arms[0]) > 0:
        return [random.choice(not_pulled_arms[0])]

    # with probability eps select random arm
    if random.random() < epsilon:
        return [random.choice(range(len(history)))]

    # else pick (one of the) best arm
    mean_scores = mean_reward(history)
    best_score = np.max(mean_scores)
    best_arms = np.nonzero(mean_scores >= (best_score - 0.00001))
    return [random.choice(best_arms[0])]
