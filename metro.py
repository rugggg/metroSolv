#Metro Solver
#Given a metro system, this program will use Q Learning to determine what the 
#best route between a given start point and end point is, given the assumption that the time value of each trip between any two stops is equal. 

import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output

#Number of metro stations
numStations = 62


#def our states, we have numStations possible states with each station
#having 1 - numStations possible actions
#row being station A row2 being at station B, row3 being at station C, and row 4 being at station D. A 0 means no direct connection, 1 means connection, 10 means action results in getting to the target station
#for now, the target station is D. We will start at A
def initState():
    state = np.zeros(numStations)
    state[0] = 1
    return state

def initStateRand():
    goal = 5
    state = np.zeros(numStations)
    station = random.randint(0,numStations-1)
    while station == goal:
        station = random.randint(0,numStations-1)
    state[station] = 1
    #state = np.array([1,0,0,0])
    return state


def initRewardM(targetStation):
    ''' 
    rM = np.array((   [ -1,  -10,   -1,  -1],
                      [-10,  10,  -10,  -1],
                      [ -1,  -10,   -1,  -1],
                      [ -1,  10,   -1,  -1]))
    '''
    ''' a hand encoded matrix    
    #             1   2   3    4   5   6   7   8   9   10
    rM = np.array(( [ -1, -1,-10,-10,-10,-10, -1,-10,-10,-10],  #1
                    [ -1, -1,-10,-10,-10,-10,-10, -1,-10,-10],  #2
                    [-10,-10, -1, -1, -1, 10,-10,-10,-10,-10],  #3
                    [-10,-10, -1, -1,-10,-10, -1,-10, -1,-10],  #4
                    [-10,-10, -1,-10, -1,-10,-10,-10, -1,-10],  #5
                    [-10,-10, -1,-10,-10, 10,-10, -1,-10,-10],  #6
                    [ -1,-10,-10, -1,-10,-10, -1,-10,-10,-10],  #7
                    [-10, -1,-10,-10,-10, 10,-10, -1,-10,-10],  #8
                    [-10,-10,-10,-10, -1,-10,-10,-10, -1, -1],  #9
                    [-10,-10,-10,-10,-10,-10,-10,-10, -1, -1],  #10
                ))    
    '''
    #ok but let's actually use a csv file, so it can be larger and more
    #editable
    rM = np.genfromtxt('londonUnderground.csv',delimiter=',')


    #now, for every -1 in the target station column, change it to a 10
    

    col = rM[:,targetStation]

    for i in range (0,len(col)):
        if col[i] == -1:
            col[i] = 10
    return rM 


def getCurrentStation(state):
    for i in range(0,numStations):
        if(state[i] == 1):
            return i;

def isValidMove(state,action,rM):
    #remember the shape of our state means the top row is
    #just the 
    curStation = getCurrentStation(state)
    if(rM[curStation,action] == -10):
        return False
    else:
#        print("true")
        return True

def makeMove(state, action, rM):
    #we will take our action, which is just an int of the station
    #we have randomly chosen to move to, check if we can move there
    #and then if so, update the one hot encoded 0 row of our state
    #to reflect our current position
    #check to see if we can make the move
    if isValidMove(state,action, rM):
        #reset our current station
        state = np.zeros(numStations)    
        state[action] = 1
    return state

def getReward(state,action,rM):
    r = rM[getCurrentStation(state), action]
 #   curStation = getCurrentStation(state)
 #   print("getting reward for: ")
 #  print("state: ", state, "location: ",curStation)
 #   r = state[curStation+1,curStation]
    if r == 10:
        return 10
    else:
        return -1

state = initState()
rM = initRewardM(2)

print("init state: ",state)
state = makeMove(state, 1, rM);
print("new state: ",state)
print("reward: ",getReward(state,1,rM))

print("next state: ",state)
state = makeMove(state, 3, rM);
print("new state: ",state)
print("reward: ",getReward(state,3,rM))

print("next state: ",state)
state = makeMove(state, 1, rM);
print("new state: ",state)
print("reward: ",getReward(state,1,rM))



#ok, now some fun Q learning things



model = Sequential()
model.add(Dense(120, init='lecun_uniform', input_shape=(numStations,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(100, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(numStations, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


# In[14]:

#model.predict(state.reshape(1,numStations), batch_size=1)
#just to show an example output; read outputs left to right: up/down/left/right

model.compile(loss='mse', optimizer=rms)#reset weights of neural network
epochs = 3000
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
#stores tuples of (S, A, R, S')
h = 0
for i in range(epochs):
    rM = initRewardM(2)
    state = initState() #using the harder state initialization function
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,numStations), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,numStations)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action, rM)
        #Observe reward
        reward = getReward(new_state, action, rM)
        
        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,numStations), batch_size=1)
                newQ = model.predict(new_state.reshape(1,numStations), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,numStations))
                y[:] = old_qval[:]
                if reward == -1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(numStations,))
                y_train.append(y.reshape(numStations,))
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = new_state
        if reward != 10: #if reached terminal state, update game status
            status = 0
        clear_output(wait=True)
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)



# In[16]:

def testAlgo():
    i = 0
    state = initState()

    print("Initial State:")
    print(state)
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,numStations), batch_size=1)
        print(qval)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action, rM)
        print(state)
        reward = getReward(state,action,rM)
        print("got reward: ",reward)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
            if reward == 10:
                print ("___________VICTORY_________")
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 100):
            print("Game lost; too many moves.")
            break
    print(rM)



# Alright, so I've empirically tested this and it trains on the easy variant with just 1000 epochs (keep in mind every epoch is a full game played to completion). Below I've implemented a function we can use to test our trained algorithm to see if it has properly learned how to play the game. It basically just uses the neural network model to calculate action-values for the current state and selects the action with the highest Q-value. It just repeats this forever until the game is won or lost. I've made it break out of this loop if it is making more than 10 moves because this probably means it hasn't learned how to win and we don't want an infinite loop running.

# In[30]:

testAlgo()
