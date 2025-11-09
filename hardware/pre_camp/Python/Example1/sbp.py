import random

argv = open('/proc/self/cmdline').read().split('\0')

def printMatrix(matrix):
	for i in range(len(matrix)):
		rowString = ""
		for j in range(len(matrix[i])):
			if i == 0:
				rowString += str(matrix[i][j]) + ","
			else:
				if (matrix[i][j] < 0) | (matrix[i][j] > 9):
					rowString += str(matrix[i][j]) + ","
				else:
					rowString += " " + str(matrix[i][j]) + ","
		print(rowString)

def matrixCreate(content):
	matrix = [0 for r in range(len(content))]
	for i in range(len(content)):
		row = content[i].split(',')
		matrix[i] = [0 for e in range(len(row) - 1)]
		for j in range(len(row) - 1):
			matrix[i][j] = int(row[j])
	return matrix

def readFile(fileName):
	with open(fileName, 'r') as file:
		return file.read().split('\n')

class BoardState:
	def __init__(self, matrix):
		self.matrix = matrix
		self.row = matrix[0][1]
		self.col = matrix[0][0]
		self.numBlocks = 0
		for i in range(2, self.row - 1):
			for j in range(1, self.col - 1):
				if (self.matrix[i][j] - 1 > self.numBlocks):
					self.numBlocks = self.matrix[i][j] - 1
	def getMatrix(self):
		return self.matrix
	def setState(self, matrix):
		self.matrix = matrix
	def getRow(self):
		return self.row
	def getColumn(self):
		return self.col
	def getPointValue(self, rowNumber: int, colNumber: int):
		return self.matrix[rowNumber][colNumber]
	def printState(self):
		printMatrix(self.matrix)
	def isDone(self):
		for i in range(1, self.row):
			if (i == 1) | (i == self.row):
				for j in self.matrix[i]:
					if j == 2:
						return True
			else:
				if (self.matrix[i][0] == 2) | (self.matrix[i][self.col - 1] == 2):
					return True
		return False
	def availableMoves(self):
		moveList = []
		for i in range(2, self.row):
			for j in range(1, self.col - 1):
				if self.matrix[i][j] == 0:
					if (self.matrix[i+1][j] > 1):
						movable = True
						for k in range(1, len(self.matrix[i+1]) - 1):
							if (self.matrix[i+1][k] == self.matrix[i+1][j]) & (self.matrix[i][k] > 0):
								movable = False
						if movable == True:
							moveList.append("(" + str(self.matrix[i+1][j]) + ",up)")
					if (self.matrix[i-1][j] > 1):
						movable = True
						for k in range(1, len(self.matrix[i-1]) - 1):
							if (self.matrix[i-1][k] == self.matrix[i-1][j]) & (self.matrix[i][k] > 0):
								movable = False
						if movable == True:
							moveList.append("(" + str(self.matrix[i-1][j]) + ",down)")
					if (self.matrix[i][j-1] > 1):
						movable = True
						for k in range(2, self.row - 1):
							if (self.matrix[k][j-1] == self.matrix[i][j-1]) & (self.matrix[k][j] > 0):
								movable = False
						if movable == True:
							moveList.append("(" + str(self.matrix[i][j-1]) + ",right)")
					if (self.matrix[i][j+1] > 1):
						movable = True
						for k in range(2, self.row - 1):
							if (self.matrix[k][j+1] == self.matrix[i][j+1]) & (self.matrix[k][j] > 0):                                                                   
								movable = False
						if movable == True:
							moveList.append("(" + str(self.matrix[i][j+1]) + ",left)")
		for i in range(1, self.col - 1):
			if (self.matrix[1][i] == -1):
				for j in range(1, len(self.matrix[2])):
					if self.matrix[2][j] == 2:
						movable = True
						for k in range(1, len(self.matrix[1])):
							if (self.matrix[2][k] == 2) & (self.matrix[1][k] != -1):
								movable = False
						if movable == True:
							moveList.append("(2,up)")
		for i in range(1, self.col - 1):
			if (self.matrix[self.row][i] == -1):
				for j in range(1, len(self.matrix[self.row - 1])):
					if self.matrix[self.row][j] == 2:
						movable = True
						for k in range(1, len(self.matrix[self.row])):
							if (self.matrix[self.row-1][k] == 2) & (self.matrix[self.row][k] != -1):
								 movable = False
						if movable == True:
							moveList.append("(2,down)")
		for i in range(2, self.row):
			if (self.matrix[i][0] == -1):
				for j in range(2, self.row - 1):
					if self.matrix[j][1] == 2:
						movable = True
						for k in range(2, self.row - 1):
							if (self.matrix[k][1] == 2) & (self.matrix[k][0] != -1):
								movable = False
						if movable == True:
							moveList.append("(2,left)")							
			if (self.matrix[i][self.col - 1] == -1):
                                for j in range(2, self.row - 1):
                                        if self.matrix[j][self.col - 2] == 2:
                                                movable = True
                                                for k in range(2, self.row - 1):
                                                        if (self.matrix[k][self.col - 2] == 2) & (self.matrix[k][self.col - 1] != -1):
                                                                movable = False
                                                if movable == True:
                                                        moveList.append("(2,right)")
		return moveList
	def move(self, blockNumber: int, direction: str):
		tempState = self.clone()
		for i in range(2, self.row):
			for j in range(1, self.col - 1):
				if blockNumber == tempState.getMatrix()[i][j]:
					if direction == "up":
						self.matrix[i-1][j] = self.matrix[i][j]
						self.matrix[i][j] = 0
					elif direction == "down":
						self.matrix[i+1][j] = self.matrix[i][j]
						self.matrix[i][j] = 0
					elif direction == "left":
						self.matrix[i][j-1] = self.matrix[i][j]
						self.matrix[i][j] = 0
					elif direction == "right":
						self.matrix[i][j+1] = self.matrix[i][j]
						self.matrix[i][j] = 0
	def clone(self):
		matrixClone = [0 for r in range(len(self.matrix))]
		for i in range(len(self.matrix)):
			matrixClone[i] = [0 for e in range(len(self.matrix[i]))]
			for j in range(len(self.matrix[i])):
				matrixClone[i][j] = self.matrix[i][j]
		clone = BoardState(matrixClone)
		return clone

class Board:
	def __init__(self, initialState: BoardState):
		self.states = [initialState]
	def printState(self, stateNumber):
		if (stateNumber >= 0) & (stateNumber < len(self.states)):
			self.states[stateNumber].printState()
	def printAvailableMoves(self):
		moves = self.states[len(self.states) - 1].availableMoves()
		for i in moves:
			print(i)
	def isDone(self):
		print(self.states[len(self.states) - 1].isDone())
	def applyMove(self, move: str):
		for i in self.states[len(self.states) - 1].availableMoves():
			if i == move:
				moveData = move.replace("(", "").replace(")", "").split(",")
				self.states.append(self.states[len(self.states) - 1].clone())
				self.states[len(self.states) - 1].move(int(moveData[0]), moveData[1])
		self.printState(len(self.states) - 1)
	def randomWalk(self, maxMoves: int):
		for i in range(maxMoves):
			if self.states[len(self.states) - 1].isDone() == True:
				break
			else:
				possibleMoves = self.states[len(self.states) - 1].availableMoves()
				randomNumber = random.randint(0, len(possibleMoves) - 1)
				move = possibleMoves[randomNumber]
				print(move)
				self.applyMove(move)
				normalizeState(self.states[len(self.states) - 1])
				print("Normalized State:")
				self.states[len(self.states) - 1].printState()
				print("*****")

def swapBlocks(state: BoardState, block1: int, block2: int):
	matrix = state.getMatrix()
	for i in range(2, state.getRow()):
		for j in range(1, state.getColumn()):
			if matrix[i][j] == block1:
				matrix[i][j] = block2
			elif matrix[i][j] == block2:
				matrix[i][j] = block1
	return matrix

def normalizeState(state: BoardState):
	matrix = state.getMatrix()
	nextBlock = 3
	for i in range(1, state.getRow()):
		for j in range(0, state.getColumn()):
			if matrix[i][j] == nextBlock:
				nextBlock += 1
			elif matrix[i][j] > nextBlock:
				matrix = swapBlocks(state, nextBlock, matrix[i][j])
				nextBlock += 1
	state.setState(matrix)

def compareStates(state1: BoardState, state2: BoardState):
	if (state1.getRow() == state2.getRow()) & (state1.getColumn() == state2.getColumn()):
		for i in range(1, state1.getRow()):
			for j in range(0, state1.getColumn()):
				if state1.getPointValue(i, j) != state2.getPointValue(i, j):
					return False
	else:
		return False
	return True

command = argv[2]
fileName = argv[3]
matrixData = readFile(fileName)
matrix = matrixCreate(matrixData)
initialState = BoardState(matrix)
board = Board(initialState)
if (command == "print"):
	board.printState(0)
elif (command == "done"):
	board.isDone()
elif (command == "availableMoves"):
	board.printAvailableMoves()
elif (command == "applyMove"):
	move = argv[4]
	board.applyMove(move)
elif (command == "compare"):
	fileName2 = argv[4]
	matrixData2 = readFile(fileName2)
	matrix2 = matrixCreate(matrixData2)
	initialState2 = BoardState(matrix2)
	print(compareStates(initialState, initialState2))
elif (command == "norm"):
	normalizeState(initialState)
	initialState.printState()
elif (command == "random"):
	maxMoves = int(argv[4])
	board.randomWalk(maxMoves)
else:
	print("Invalid Command")
