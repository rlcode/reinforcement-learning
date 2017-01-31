import copy


class ValueIteration():

    def __init__(self,width=5,height=5, discount=0.9,iteration=10):
        self.discount = discount
        self.iteration = iteration
        self.width = width
        self.height = height
        self.Qvalues = [[0.00] * 5 for i in range(5)]
        self.reward = [[0] * 5 for i in range(5)]
        self.states = []
        self.possible_actions = [(0,1),(0,-1),(-1,0),(1,0)] # (-1,0) 상 (1,0) 하 (0,-1) 좌 (0,1)우
        self.policies = [[self.possible_actions] * 5 for i in range(5)]
        self.transitionProb = 1
        self.timeStepReward = 0
        self.reward[2][2] = 1
        self.policies[2][2] = []
        self.reward[1][2] = -1
        self.reward[2][1] = -1
        self.QvalueCopy = self.Qvalues.copy()


        for x in range(width):
            for y in range(height):
                state = [x,y]
                self.states.append(state)


    def do_iteration(self, number):
        QvaluesCopy = copy.deepcopy(self.Qvalues)
        for i in range(number):
            for state in self.states:
                finalValue = 0
                len_action = len(self.policies[state[0]][state[1]])
                for action in self.policies[state[0]][state[1]]:
                    currentValue = self.computeQValueFromValues(state,action)/len_action
                    if currentValue>finalValue:
                        finalValue = currentValue
                QvaluesCopy[state[0]][state[1]] = round(finalValue,2)
            self.Qvalues = copy.deepcopy(QvaluesCopy)

    def state_sum(self,state,action):
        state_ = [state[0]+action[0],state[1]+action[1]]
        if state_[0] < 0:
            state_[0] = 0
        if state_[1] < 0:
            state_[1] = 0
        if state_[0] > 4:
            state_[0] = 4
        if state_[1] > 4:
            state_[1] = 4
        return state_

    def getValue(self, state):
        if state[0]>4:
            return round(self.Qvalues[4][state[1]],2)
        if state[1]>4:
            return round(self.Qvalues[state[0]][4],2)
        if state[0]<0:
            return round(self.Qvalues[0][state[1]],2)
        if state[1]<0:
            return round(self.Qvalues[state[0]][0],2)
        return round(self.Qvalues[state[0]][state[1]],2)

    def computeQValueFromValues(self, state, action):
        state_ = self.state_sum(state,action)
        if state_[0] < 0:
            state_[0] = 0
        if state_[1] < 0:
            state_[1] = 0
        if state_[0] > 4:
            state_[0] = 4
        if state_[1] > 4:
            state_[1] = 4

        if state[0]==2 and state[1]==2:
            return 0

        value = self.getValue(state_)
        value *= self.transitionProb
        value *= self.discount
        value += self.reward[state_[0]][state_[1]]
        value += self.timeStepReward
        return value

    def computeActionFromValues(self, state):

        possibleActions = self.possible_actions

        if len(possibleActions) == 0:
            return None

        value = -99999
        result = []
        for action in possibleActions:
            state_ = self.state_sum(state,action)
            temp = self.computeQValueFromValues(state, action)
            temp += self.reward[state_[0]][state_[1]]

            if temp == value:
                result.append(action)
            elif temp > value:
                value = temp
                result =[]
                result.append(action)

        return result

    def updatePolicy(self):
        for i in range(self.width):
            for j in range(self.height):
                if i==2 and j ==2:
                    continue
                self.policies[i][j] = self.computeActionFromValues((i, j))

    def getPolicies(self):
        return copy.deepcopy(self.policies)

    def getValues(self):
        return copy.deepcopy(self.Qvalues)

if __name__ == "__main__":
    agent = ValueIteration()
    agent.do_iteration(10)
    agent.updatePolicy()
    print(agent.getValues())

