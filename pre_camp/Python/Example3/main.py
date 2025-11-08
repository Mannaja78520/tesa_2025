import agent
import othello
import game
import sys
import time
import pp686

def create_player(arg,  depht_or_time):
    if arg == 'human':
        return agent.HumanPlayer()
    elif arg == 'random':
        return agent.RandomAgent()
    elif arg == 'minimax':
        return agent.MinimaxAgent(int(depht_or_time))
    elif arg == 'alphabeta':
        return agent.AlphaBeta(int(depht_or_time))
    elif arg == 'extra':
        return pp686.extra(depht_or_time)
    else:
        agent.RandomAgent()

def get_arg(index, default=None):
    '''Returns the command-line argument, or the default if not provided'''
    return sys.argv[index] if len(sys.argv) > index else default

if __name__ == '__main__':

    initial_state = othello.State()

    if len(sys.argv) > 1:
        agent1 = sys.argv[1]
        agent2 = sys.argv[2]
        depht_or_time = 3
    if len(sys.argv) == 4 : 
        depht_or_time = float(sys.argv[3])        


    player1 = create_player(get_arg(1), depht_or_time)
    player2 = create_player(get_arg(2), depht_or_time)

    # player1 = agent.HumanPlayer()
    # player2 = agent.RandomAgent()

    game = game.Game(initial_state, player1, player2)
    startTime = time.time()
    game.play()
    stopTime = time.time()
    print("Total time: " + str(stopTime - startTime))

    
