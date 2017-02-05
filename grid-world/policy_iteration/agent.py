import copy


class PolicyIterationAgent():

    def __init__(self,width=5,height=5, discount=0.9):
        self.discount = discount # discount factor
        self.width = width # grid world width
        self.height = height #grid world height
        self.values = [[0.00] * 5 for i in range(5)] # 2D values array
        self.reward = [[0] * 5 for i in range(5)] # 2D rewards array
        self.states = [] #states array indices
        self.possibleActions = [(0,1),(0,-1),(-1,0),(1,0)] # (-1,0)상 (1,0)하 (0,-1)좌 (0,1)우
        self.policies = [[self.possibleActions] * 5 for i in range(5)]  #2D policies array and initialization with possible actions
        self.transitionProb = 1 # transition probability,  set to 1
        self.timeStepReward = 0 # time step reward, set to 0
        self.reward[2][2] = 1 # reward for fish
        self.policies[2][2] = []
        self.reward[1][2] = -1 # reward for fire
        self.reward[2][1] = -1 # reward for fire
        self.valueCopy = self.values.copy()


        for x in range(width): #states array initialization
            for y in range(height):
                state = [x,y]
                self.states.append(state)


    def doIteration(self, number): # doing iteration
        valuesCopy = copy.deepcopy(self.values) #copy values for calculation
        for i in range(number):  #iteration for given number times
            for state in self.states:
                finalValue = 0
                actionLength = len(self.policies[state[0]][state[1]]) # possible action number
                for action in self.policies[state[0]][state[1]]:
                    currentValue = self.computeQValueFromValues(state,action)/actionLength #probability for each action = values / possible action number
                    finalValue += currentValue
                valuesCopy[state[0]][state[1]] = round(finalValue,2)
            self.values = copy.deepcopy(valuesCopy)

    def stateActionSum(self,state,action): # state after action,  param : state, action
        tempSum = [state[0]+action[0],state[1]+action[1]]
        if tempSum[0] < 0: # Adjust to not exceed the boundaries
            tempSum[0] = 0
        if tempSum[1] < 0:
            tempSum[1] = 0
        if tempSum[0] > 4:
            tempSum[0] = 4
        if tempSum[1] > 4:
            tempSum[1] = 4
        return tempSum

    def getValue(self, state): # Returns the value of a particular state
        if state[0]>4: # Adjust to not exceed the boundaries
            return round(self.values[4][state[1]],2)
        if state[1]>4:
            return round(self.values[state[0]][4],2)
        if state[0]<0:
            return round(self.values[0][state[1]],2)
        if state[1]<0:
            return round(self.values[state[0]][0],2)
        return round(self.values[state[0]][state[1]],2)

    def computeQValueFromValues(self, state, action): #compute Qvalue from state and action
        stateTemp = self.stateActionSum(state,action)
        if stateTemp[0] < 0:
            stateTemp[0] = 0
        if stateTemp[1] < 0:
            stateTemp[1] = 0
        if stateTemp[0] > 4:
            stateTemp[0] = 4
        if stateTemp[1] > 4:
            stateTemp[1] = 4

        if state[0]==2 and state[1]==2:
            return 0

        value = self.getValue(stateTemp)
        value *= self.transitionProb
        value *= self.discount
        value += self.reward[stateTemp[0]][stateTemp[1]]
        value += self.timeStepReward
        return value

    def computeActionFromValues(self, state): #compute action from values : choose policies

        possibleActions = self.possibleActions

        if len(possibleActions) == 0:
            return None

        value = -99999
        result = []
        for action in possibleActions:
            stateTemp = self.stateActionSum(state,action)
            temp = self.computeQValueFromValues(state, action)
            temp += self.reward[stateTemp[0]][stateTemp[1]]
            if temp == value:
                result.append(action)
            elif temp > value:
                value = temp
                result =[]
                result.append(action)
        return result

    def updatePolicy(self): #update policy for every state
        for i in range(self.width):
            for j in range(self.height):
                if i==2 and j ==2:
                    continue
                self.policies[i][j] = self.computeActionFromValues((i, j))

    def getPolicies(self): #get whole policies
        return copy.deepcopy(self.policies)

    def getValues(self): # get whole values array
        return copy.deepcopy(self.values)
