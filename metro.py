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
numStations = 7 #this is for the small set
#numStations = 62 #this is for the london underground

goal = 6
#def our states, we have numStations possible states with each station
#having 1 - numStations possible actions
#row being station A row2 being at station B, row3 being at station C, and row 4 being at station D. A 0 means no direct connection, 1 means connection, 10 means action results in getting to the target station
#for now, the target station is D. We will start at A
def initState():
    state = np.zeros((2,numStations))
    state[0,3] = 1 #place player at station n
    state[1,goal] = 1 #place goal
    return state

def initStateC(start,goal):
    state = np.zeros((2,numStations))
    state[0,start] = 1 #place player at station 1 for simplicity
    state[1,goal] = 1 #place goal
    return state


def initStateRand():
    state = np.zeros((2,numStations))
    station = random.randint(0,numStations-1)
    while station == goal:
        station = random.randint(0,numStations-1)
    state[0,station] = 1
    state[1,goal] = 1
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
    #rM = np.genfromtxt('londonUnderground.csv',delimiter=',')
    rM = np.genfromtxt('small.csv',delimiter=',')
    
    #now, for every -1 in the target station column, change it to a 10

    col = rM[:,targetStation]
    row = rM[targetStation,:]
    

    for i in range (0,len(col)):
         if col[i] == -1:
            col[i] = 10
         if row[i] == -1:
            row[i] = 10
    return rM 

def getCurrentStation(state):
    for i in range(0,numStations):
        if(state[0,i] == 1):
            return i;

def isValidMove(state,action,rM):
    #remember the shape of our state means the top row is
    #just the 
    curStation = getCurrentStation(state)
    if(rM[curStation,action] == -10):
        return False
    else:
        return True

def makeMove(state, action, rM):
    #we will take our action, which is just an int of the station
    #we have randomly chosen to move to, check if we can move there
    #and then if so, update the one hot encoded 0 row of our state
    #to reflect our current position
    #check to see if we can make the move
    if isValidMove(state,action, rM):
        #reset our current station
        state[0] = np.zeros(numStations)    
        state[0,action] = 1
    return state

def getReward(state,rM):
    print("getting reward for: \n",state)
    station = getCurrentStation(state);
    print("Station: ",station);
    r = rM[station,station];
    print("Reward: ",r)
    return r
 #   curStation = getCurrentStation(state)
 #   print("getting reward for: ")
 #  print("state: ", state, "location: ",curStation)
 #   r = state[curStation+1,curStation]
 #   if r == 10:
 #       return 10
 #   else:
 #       return -1


st = initStateC(5,3)
rLL = initRewardM(3)

getReward(st,rLL)

def getRandomAction(state,rM):
    #get all items in the row that are
    index = getCurrentStation(state)
    actionSpace = []
    possible = rM[:,index]
    for i in range(0,len(possible)):
        if(possible[i] != -10 and i != index):
            actionSpace.append(i)
    return random.choice(actionSpace)

def getBestValidAction(state,prev,qval,rM):
    print("*********************")
    index = getCurrentStation(state)
    print("Getting best valid action for: ",index)
    actionSpace = []
    possible = rM[index,:]
    print("possible: ",possible)
    for i in range(0,len(possible)):
        if(possible[i] != -10 and i != index and i != prev):
            actionSpace.append(i)
    if(len(actionSpace) < 1):
        #print("=//////////////////////")
        print("Problem isolated node")
        #print("=//////////////////////")
        for i in range(0,len(possible)):
            if(possible[i] != -10 and i != index):
                actionSpace.append(i)
    
    print("Action Space: ",actionSpace)
    #Ok, we have our action space of possible nodes to move to
    #now, we need to get the Q value for each one, and return 
    #what move to make based on that

    best = 0 #arbitrarily small number 
    bestQ = 0
    print("comp to QVAL: ",qval)
    for i in range(0, len(actionSpace)):
        print("---Action Space Val: ",actionSpace[i])
        print("---QVal: ",qval[0:,actionSpace[i]])
        if(qval[0:,actionSpace[i]] > bestQ or bestQ == 0):
            best = actionSpace[i]
            bestQ = qval[0:,best]
    print("Best: ",best)
    print("Best QVal: ",qval[0:,best])
    print("********************")
    return best



model = Sequential()
model.add(Dense(220, init='lecun_uniform', input_shape=(numStations*2,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(numStations, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


model.compile(loss='mse', optimizer=rms)#reset weights of neural network
epochs = 100
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
h = 0
tTot = 0
for i in range(epochs):
    print("\n")
    print("Init New Game \n")
    rM = initRewardM(goal)
    print("RM::::\n")
    print(rM)
    #state = initState() #using the deterministic
    state = initStateRand() #using the random start
    prev = getCurrentStation(state)
    status = 1
    #while game still in progress
    t = 0
    randMoves = 0
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print("&&&&     GAME ",i,"        &&")
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        #print("+++++++++++++++++++++++++++++++++++++++++")
        qval = model.predict(state.reshape(1,numStations*2), batch_size=1)
        if (random.random() < epsilon): #choose random action
            #action = np.random.randint(0,numStations)
            #action = getBestValidAction(state,rM)
            randMoves += 1           
            action = getRandomAction(state,rM)
        else: #choose best action from Q(s,a) values
            #get indexs of possible actions
            #select max of those
            action = getBestValidAction(state,prev,qval,rM)
            #action = (np.argmax(qval))
        #Take action, observe new state S'
        print("take action: ",action)
        
        prev = getCurrentStation(state)
        new_state = makeMove(state, action, rM)
        reward = getReward(new_state, rM)

        
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
                old_qval = model.predict(old_state.reshape(1,numStations*2), batch_size=1)
                newQ = model.predict(new_state.reshape(1,numStations*2), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,numStations))
                y[:] = old_qval[:]
                if reward == -1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(numStations*2,))
                y_train.append(y.reshape(numStations,))
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = new_state
            reward = getReward(state,rM)
            print("---_----__Reward Check: ",reward)
            print(state)
            
        t += 1
        if reward == 10: # or reward == 10.0: #if reached terminal state, update game status
            status = 0
            print("VICTORY")
            print("Num Turns: ",t)
            print("Rand Moves: ",randMoves)
            tTot += t
        clear_output(wait=True)
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("t total: ",tTot)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

losess = 0 

wins = 0 


def testAlgo():
    i = 0
    state = initState()
    state = initStateRand()

    print("Initial State:")
    print(state)

    prev = getCurrentStation(state)
    status = 1
    #while game still in progress
    while(status == 1):
        #print("((((((((()))))))")
        #print("((((((((()))))))")
        #print("((((((((()))))))")
        #print("      Turn: ",i)
        qval = model.predict(state.reshape(1,numStations*2), batch_size=1)
        print(qval)
        #action = (np.argmax(qval)) #take action with highest Q-value
        action = getBestValidAction(state,prev,qval,rM)
        
        #take action with highest Q-value

        prev = getCurrentStation(state)
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action, rM)
        print(state)
        reward = getReward(state,rM)
        #reward = rM[getCurrentStation(state), getCurrentStation(state)]
        print("got reward: ",reward)
        if reward == 10:
            status = 0
            print("Reward: %s" % (reward,))
            print ("___________VICTORY_________")
            return 1
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 60):
            print("Game lost; too many moves.")
            print("______FAILURE_____");
            return 0
            break
total = 0
numTests = 100
for i in range(0,numTests):
    total += testAlgo()

print("Wins: ",total)
print("Losses: ",numTests-total)
