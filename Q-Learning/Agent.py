from random import randint
import numpy as np
from pandas import DataFrame

#the Agent Class is used for test the trained agent
class Agent():

    def __init__(self):
        self.name = "RLAgent"
        self.Qtable = None #the Q-table
    
    #change the stateObs to the state we use (almost a 5x5 string)
    def getState(self, stateObs):
        for index in range(10):
            if sum(stateObs[85][10*index+5]) == sum([156, 156, 156, 255]):
               break
        string = ''
        state = 0
        for i in range(4,9):
          for j in range(max(0,index-2),min(10,index+3)):
          #for j in range(0,10):
            s = sum(stateObs[10*i+5][10*j+5]) + sum(stateObs[10*i+5][10*j+4])
            
            if s == 1446:
               state = 1 #airPlane
            elif s == 738:
               state = 2 #alien
            elif s == 1863:
               state = 3 #bomb
            else:
               state = 0
            string += str(state)
        return string
        
    #the act function
    def act(self, stateObs, actions):
        state = self.getState(stateObs)  # get current state
        if not state in self.Qtable:  # find it in the Qtable, if not, use the [0,0,0,0]
             self.Qtable[state] = [0,0,0,0]
        if self.Qtable[state] == [0,0,0,0]:
             action_id = randint(0,len(actions)-1)  # if it is a new state, we will choose random action
        else:
             action_id = self.Qtable[state].index(max(self.Qtable[state]))  # choose the action that has the max reward in Q-table
        return action_id
        
#the trainAgent Class is used for train the trained agent
class trainAgent():

    def __init__(self):
        self.name = "Q-learningTrainAgent"  #trainAgentName
        self.epsilon = 1                    #random exploration rate
        self.Qtable = None                  #Q-table
        self.beta = 0.05                    #beta shows how epsilon decreased
        self.lastState = None               #every time we renew the Q-table, we must know last state, so it is the last stste
    
    #this function is used to inital Qtable for the first time and initial the "lastState"
    def inital(self, stateObs):
        self.lastState = self.getState(stateObs)
        self.Qtable = {self.lastState:[0,0,0,0]}
    
    #this function is the main function for Q-learning
    def trainQlearning(self, stateObs, increScore, action_id):
        state = self.getState(stateObs) #get current state
        # cmpute the reward by increScore
        if increScore == -1:
           reward = -80
        elif increScore == 0:
           reward = -0.2
        elif increScore == 2:
           reward = 20
        else:
           reward = 5
           
        # add the state for the first time
        if not (self.lastState in self.Qtable):
           self.Qtable[self.lastState] = [0,0,0,0]
        
        # add the state for the first time
        if not (state in self.Qtable):
           self.Qtable[state] = [0,0,0,0]
        
        # renew the Q-table
        self.Qtable[self.lastState][action_id] = self.Qtable[self.lastState][action_id] + 0.5*(reward + 0.9*max(self.Qtable[state]) - self.Qtable[self.lastState][action_id])
        
        # renew the state
        self.lastState  = state
        
    
    #change the stateObs to the state we use (almost a 5x5 string)
    def getState(self, stateObs):
      for index in range(10):
          if sum(stateObs[85][10*index+5]) == sum([156, 156, 156, 255]):
             break
      string = ''
      state = 0
      for i in range(4,9):
        for j in range(max(0,index-2),min(10,index+3)):
          #compute s and then we know what it is in this 10x10
          s = sum(stateObs[10*i+5][10*j+5]) + sum(stateObs[10*i+5][10*j+4])
          
          if s == 1446:
             state = 1 #airPlane
          elif s == 738:
             state = 2 #alien
          elif s == 1863:
             state = 3 #bomb
          else:
             state = 0 #others
          string += str(state)
      return string
    
    #the act function
    def act(self, t, stateObs, actions):
        # renew the epsilon
        self.epsilon = np.exp(-self.beta * t)
        
        # get the current state
        state = self.lastState
        
        # if epsilon is large, we may choose random action
        if np.random.rand() < self.epsilon:
            action_id = randint(0,len(actions)-1)
        else:
            if not state in self.Qtable: 
               self.Qtable[state] = [0,0,0,0] #add new state
            if self.Qtable[state] == [0,0,0,0]: 
               action_id = randint(0,len(actions)-1) # if it is a new state, we will choose random action
            else:
               action_id = self.Qtable[state].index(max(self.Qtable[state])) # choose the action that has the max reward in Q-table
        return action_id
        
