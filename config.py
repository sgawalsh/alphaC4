from math import sqrt

#contains config values that determine several import behavious of MCTS and selfplay learning functions

#MCTSexploration = sqrt(2) #standard value which determines how likely a node is to be explored, given how many times it has been explored vs the rest of the tree (larger value means more likely to select unexplored nodes), results in overly greedy play for connect 4
MCTSexploration = .5 #that's better
maxBoardVal = 100
miniMaxDefaultDepth = 4
trainingRecursionCount = 50 # how many selection cycles used in MCTS
trainingSetSize = 50 # how many games are added to training set by champion each iteration
fullTrainingSetSize = 1000 # maximum games in training set
challengerSamples = 2000 # how many boards new challenger is trained on
showDownSize = 50 # how many games played between challenger and champion
selfTournamentShowDownSize = 5 # games played between each generation in selfplay tournament
winRatio = 55 / 45 # ratio of wins to losses to become new champion
champArrayLength = 30 # number of former champions recorded

class node():
	def __init__(self, board, isRedTurn, parent, rowNum, colNum):
		self.parent = parent
		self.isRedTurn = isRedTurn
		self.children = []
		self.board = board
		self.rowNum = rowNum
		self.colNum = colNum
	
def maxelements(seq):#from https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list
    #Return list of position(s) of largest element
    max_indices = []
    if seq:
        max_val = seq[0]
        for i,val in ((i,val) for i,val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i)
            else:
                max_val = val
                max_indices = [i]

    return max_indices
	
def getResponseFromList(question, answers):
	while True:
		data = input(question + " " + str(answers) + "\n> ")
		if data:
			if data in answers:
				return data
			else:
				print("Please choose an option from the list")

def getIntResponse(question):
	while True:
		try:
			i = int(input(question + "\n> "))
			if i <= 0:
				print("Value out of range")
			else:
				return i
		except ValueError:
			print("Please enter an integer.")