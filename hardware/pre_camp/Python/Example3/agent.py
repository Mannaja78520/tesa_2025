import math
import random

import game

class HumanPlayer(game.Player):

	def __init__(self):
		super().__init__()

	def choose_move(self, state):
        # generate the list of moves:
		moves = state.generateMoves()

		for i, action in enumerate(moves):
			print('{}: {}'.format(i, action))
		response = input('Please choose a move: ')
		return moves[int(response)]

class RandomAgent(game.Player):
	def __init__(self):
		super().__init__()

	def choose_move(self, state):
		moves = state.generateMoves()
		numMoves = len(moves)
		if numMoves > 0:
			index = random.randint(0, len(moves) - 1)
			return moves[index]
		return None


class MinimaxAgent(game.Player):
	def __init__(self, maxDepth):
		super().__init__()
		self.maxDepth = maxDepth

	def choose_move(self, state):
		bestMove = None
		bestV = -math.inf
		for i in state.generateMoves():
			depthCount = 0
			v = self.minValue(state.applyMoveCloning(i), depthCount)
			if v > bestV:
		    		bestV = v
		    		bestMove = i
		return bestMove

	def maxValue(self, state, depthCount):
		if state.game_over() or (depthCount >= self.maxDepth):
			return state.score()
		v = -math.inf #initial value
		for i in state.generateMoves():
			currentDepth = depthCount + 1
			v = max(v, self.minValue(state.applyMoveCloning(i), currentDepth))
		return v


	def minValue(self, state, depthCount):
		if state.game_over()  or (depthCount >= self.maxDepth):
			return state.score()
		v = math.inf #initial value
		for i in state.generateMoves():
			currentDepth = depthCount + 1
			v = min(v, self.maxValue(state.applyMoveCloning(i), currentDepth))
		return v

class AlphaBeta(game.Player):
	def __init__(self, maxDepth):
		super().__init__()
		self.maxDepth = maxDepth
		self.a = -math.inf
		self.b = math.inf

	def choose_move(self, state):
		bestMove = None
		bestV = -math.inf

		for i in state.generateMoves():
			depthCount = 0
			v = self.minValue(state.applyMoveCloning(i), depthCount)
			if v > bestV:
				bestV = v
				bestMove = i
		#reset alpha and beta for the next turn
		self.a = -math.inf
		self.b = math.inf
		return bestMove

	def maxValue(self, state, depthCount):
		if state.game_over() or (depthCount >= self.maxDepth):
			return state.score()
		v = -math.inf #initial value
		for i in state.generateMoves():
			currentDepth = depthCount + 1
			v = max(v, self.minValue(state.applyMoveCloning(i), currentDepth))
			if v >= self.b:
				return v
			self.a = max(self.a, v)
		return v
	
	def minValue(self, state, depthCount):
		if state.game_over()  or (depthCount >= self.maxDepth):
			return state.score()
		v = math.inf #initial value
		for i in state.generateMoves():
			currentDepth = depthCount + 1
			v = min(v, self.maxValue(state.applyMoveCloning(i), currentDepth))
			if v <= self.a:
				return v
			self.b = min(self.b, v)
		return v
