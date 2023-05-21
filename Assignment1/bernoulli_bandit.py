
import numpy as np
import matplotlib.pyplot as plt

class Bernoullibandit():
    """
    This class creates a Bernoulli bandit simulator 
    Arguments:
            probability of each arm's sucess+
    return:
            It gives rewards as per input of the probability of each arm"""
    def __init__(self,p,p_i):
        self.p = p
        self.p_i = p_i
        self.k = len(p)
        self.est_p = [self.p_i for _ in range(self.k)]
        self.n = np.zeros(self.k)
        self.action_values = np.zeros(self.k)
        self.best_action = np.argmax(p)
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)        

    def sample(self, action):
        return np.random.binomial(1, self.p[action])
    
    def update(self, action, reward, alpha):
        # self.n[action] += 1
        self.est_p[action] = self.est_p[action] + alpha*(reward - self.est_p[action])
    
    def updateAvg(self, action, reward):
        self.n[action] += 1
        self.est_p[action] += (reward - self.est_p[action])/self.n[action]
    
    def epsilon_greedy(self, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.est_p)

    def ucb(self, c, t):
        if t ==0:
            return np.random.randint(0, self.k)
        else:    
            ucb = [self.est_p[i] + c*np.sqrt(np.log(t+1)/(self.n[i]+1)) for i in range(self.k)]
            action = np.argmax(ucb)
            # self.n[action] += 1
            return action
    
    def std_error(self, action, rewad, j):
        return ((rewad - self.est_p[action])/j)

    def thomson_sampling(self):
        theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(theta)
        #update after the sample part.

    def update_thomson(self, action, reward):
        self.alpha[action] += reward
        self.beta[action] += 1-reward 
        
    def first_fraction(self, num_runs, num_timestep, best_action):
        # best_action = np.ones((num_runs, num_timestep))*(-1)
        first_fraction = []
        for j in range(num_timestep):
            first_action = 0
            for i in range(num_runs):
                if best_action[i,j] == 0.0:
                    first_action += 1
            first_fraction.append((first_action/num_runs))
        return first_fraction
    

    