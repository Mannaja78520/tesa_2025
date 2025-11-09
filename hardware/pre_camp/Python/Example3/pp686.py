import math
import random
import time
import game

class extra(game.Player):
	def __init__(self, maxTime):
		super().__init__()
		if maxTime < 0.1:
			self.maxTime = 0.1 # maxTime >= 0.1s or 100ms
		else:
			self.maxTime = maxTime
		self.a = -math.inf
		self.b = math.inf
		self.startTime = time.time()

	def choose_move(self, state):
		bestMove = None
		bestV = -math.inf

		for i in state.generateMoves():
			v = self.minValue(state.applyMoveCloning(i), time.time())
			if v > bestV:
				bestV = v
				bestMove = i
		#reset alpha and beta for the next turn
		self.a = -math.inf
		self.b = math.inf
		return bestMove

	def maxValue(self, state, startTime):
		if state.game_over() or ((time.time() - startTime) >= self.maxTime):
			return state.score()
		v = -math.inf #initial value
		for i in state.generateMoves():
			v = max(v, self.minValue(state.applyMoveCloning(i), startTime))
			if v >= self.b:
				return v
			self.a = max(self.a, v)
		return v
	
	def minValue(self, state, startTime):
		if state.game_over()  or ((time.time() - startTime) >= self.maxTime):
			return state.score()
		v = math.inf #initial value
		for i in state.generateMoves():
			v = min(v, self.maxValue(state.applyMoveCloning(i), startTime))
			if v <= self.a:
				return v
			self.b = min(self.b, v)
		return v
	
	def timeElapsed(self):
		return time.time() - self.startTime

	def timeReset(self):
		self.startTime = time.time()
