#!/usr/bin/env python
import gym
import gym_gvgai
import Agent as Agent
import json

env = gym_gvgai.make('gvgai-aai-lvl0-v0') #load the game
agent = Agent.Agent() #build the agent object
print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
# reset environment
stateObs = env.reset()
actions = env.env.GVGAI.actions()  #get the actions

#############this is new to load the model#########
with open("model.json",'r',encoding='utf-8') as json_file:
     # load the model
     agent.Qtable=json.load(json_file)
#############this is new to load the model#########     

#############this is new to compute the total score#########
Score = 0
#############this is new to compute the total score#########

for t in range(1000):

    # choose action based on trained policy
    action_id = agent.act(stateObs,actions)

    # do action and get new state and its reward
    stateObs, increScore, done, debug = env.step(action_id)
    
    print("Action " + str(action_id) + " tick " + str(t+1) + " reward " + str(increScore) + " win " + debug["winner"])
    
    
    #############this is new to compute the total score#########
    Score += increScore #get the total score
    #############this is new to compute the total score#########
    
    # break loop when terminal state is reached
    if done:
        break

#############this is new to print the total score, win or not, tick#########
print(Score, debug["winner"], t)
#############this is new to print the total score, win or not, tick#########