import gym
import gym_gvgai
import Agent as Agent
import json
import datetime

#just as testRLAgent.py do to load the game and reset the environment
env = gym_gvgai.make('gvgai-aai-lvl0-v0')

#build a trainAgent object
train_agent = Agent.trainAgent()
print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))

# reset environment
stateObs = env.reset()

#get the actions
actions = env.env.GVGAI.actions()

#inital the Qtable for the first time and inital "lastState"
train_agent.inital(stateObs)

#read the model from a json file
with open("model.json",'r',encoding='utf-8') as json_file:
     train_agent.Qtable=json.load(json_file)
     
print("start training...")
#This is just for recording time
start = datetime.datetime.now()

for i in range(10):
    #every time we should reset the environment
    stateObs = env.reset()
    
    #the total Score
    Score = 0
    for t in range(1000):
    
        # choose action based on trained policy
        action_id = train_agent.act(t,stateObs,actions)

        # do action and get new state and its reward
        stateObs, increScore, done, debug = env.step(action_id)
        #print("Action " + str(action_id) + " tick " + str(t+1) + " reward " + str(increScore) + " win " + debug["winner"])
        
        # caculate the total score
        Score += increScore

        # train the model
        train_agent.trainQlearning(stateObs, increScore, action_id)

        # break loop when terminal state is reached
        if done:
            break
            
    #print the total Score        
    print(Score,debug["winner"],t)

#This is also for recording time
end = datetime.datetime.now()
print("training finished!")
print ("training time: " + str(end-start))

#write the model to the json file
print(len(train_agent.Qtable))
with open("model.json",'w',encoding='utf-8') as json_file:
    json.dump(train_agent.Qtable,json_file,ensure_ascii=False)
    