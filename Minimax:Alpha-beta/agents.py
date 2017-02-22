import random
import sys
from game import pickup_game, PickupState
import numpy as np
from copy import copy, deepcopy

class GamePlayer(object):
    '''Represents the logic for an individual player in the game'''

    def __init__(self, player_id, game):
        '''"player_id" indicates which player is represented (int)
        "game" is a game object with a get_successors function'''
        self.player_id = player_id
        self.game = game
        return

    def evaluate(self, state):
        '''Evaluates a given state for the specified agent
        "state" is a game state object'''
        pass

    def minimax_move(self, state):
        '''Returns a string action representing a move for the agent to make'''
        pass

    def alpha_beta_move(self, state):
        '''Same as minimax_move with alpha-beta pruning'''
        pass


class BasicPlayer(GamePlayer):
    '''A basic agent which takes random (valid) actions'''

    def __init__(self, player_id, game):
        GamePlayer.__init__(self, player_id, game)

    def evaluate(self, state):
        '''This agent doesn't evaluate states, so just return 0'''
        return 0

    def minimax_move(self, state):
        '''Don't perform any game-tree expansions, just pick a random move
            that's available in the list of successors'''
        assert state.player == self.player_id
        successors, actions = self.game.get_successors(state)
        # Take a random successor's action
        return random.choice(actions)

    def alpha_beta_move(self, state):
        '''Just calls minimax_move'''
        return self.minimax_move(state)


countMinimax = 0
countAlpha = 0

def minimax_dfs(game, state, depth, horizon, eval_fn):
    global countMinimax
    """Return (value, action) tuple for minimax search up to the given depth"""
    # *** YOUR CODE HERE ***
    # Note that eval_fn is a function which has been passed as an argument and
    # you can call it like any other function
    if depth > horizon or state.cookiecounts[0] + state.cookiecounts[1] == 13:
        return (eval_fn(state), 'z')

    bestValue = 0
    bestAction = 'none'
    dirs = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]
    acts = ['n', 's', 'e', 'w', 'z']
    # If maximizing player
    if depth %2 == 0:
        bestValue = -sys.maxint
        successors, actions = game.get_successors(state)
        for index, child in enumerate(successors):
            v, action = minimax_dfs(game, child, depth+1, horizon, eval_fn)
            countMinimax = countMinimax + 1
            if v > bestValue:
                bestValue = v
                bestAction = actions[index]
    elif depth %2 == 1:
        bestValue = sys.maxint
        successors, actions = game.get_successors(state)
        for index, child in enumerate(successors):
            v, action = minimax_dfs(game, child, depth+1, horizon, eval_fn)
            countMinimax = countMinimax + 1
            if v < bestValue:
                bestValue = v
                bestAction = actions[index]
    #print bestValue
    return (bestValue, bestAction)
    raise NotImplementedError()

def alphabeta_dfs(game, state, depth, horizon, eval_fn, alpha, beta):
    global countAlpha
    if depth > horizon or state.cookiecounts[0] + state.cookiecounts[1] == 13:
        return (eval_fn(state), 'z')

    bestValue = 0
    bestAction = 'none'
    dirs = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]
    acts = ['n', 's', 'e', 'w', 'z']
    # If maximizing player
    if depth %2 == 0:
        bestValue = -sys.maxint
        successors, actions = game.get_successors(state)
        for index, child in enumerate(successors):
            v, action = alphabeta_dfs(game, child, depth+1, horizon, eval_fn, alpha, beta)
            countAlpha = countAlpha + 1
            if v > bestValue:
                bestValue = v
                alpha = max(alpha, bestValue)
                bestAction = actions[index]
                if beta <= alpha:
                    break
    elif depth %2 == 1:
        bestValue = sys.maxint
        successors, actions = game.get_successors(state)
        for index, child in enumerate(successors):
            v, action = alphabeta_dfs(game, child, depth+1, horizon, eval_fn, alpha, beta)
            countAlpha = countAlpha + 1
            if v < bestValue:
                bestValue = v
                beta = min(beta, bestValue)
                bestAction = actions[index]
                if beta <= alpha:
                    break
    #print bestValue
    return (bestValue, bestAction)

def isValid(state, row, col, grid):
    if (row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0])):
        return False
    return True

# Returns location of player 0 and 1 as an array of tuples
def findPlayer(state):
    grid = state.grid
    loc0 = [0, 0]
    loc1 = [0, 0]
    for i in range(3):
        for j in range(5):
            if grid[i][j] == '0':
                loc0[0] = i
                loc0[1] = j
            if grid[i][j] == '1':
                loc1[0] = i
                loc1[1] = j
    return [loc0, loc1]


class StudentPlayer(GamePlayer):
    def __init__(self, player_id, game):
        GamePlayer.__init__(self, player_id, game)

    def evaluate(self, state):
        # *** YOUR CODE HERE ***
        # Losing state
        if state.cookiecounts[0] + state.cookiecounts[1] == 13:
            if state.player == 0 and state.cookiecounts[0] < state.cookiecounts[1]:
                return -sys.maxint
            elif state.player == 1 and state.cookiecounts[1] < state.cookiecounts[0]:
                return -sys.maxint
        myLoc = [0,0]
        opponentLoc = [0,0]

        myValue = 0
        opponentValue = 0

        # Find my location
        if state.player == 0:
            for i in range(3):
                for j in range(5):
                    if state.grid[i][j] == '0':
                        myLoc[0] = i
                        myLoc[1] = j
                    if state.grid[i][j] == '1':
                        opponentLoc[0] = i
                        opponentLoc[1] = j
            # Manhatten distance
            for i in range(3):
                for j in range(5):
                    if state.grid[i][j] == 'c':
                        myValue = myValue + abs(myLoc[0]-i) + abs(myLoc[1]-j)
                        opponentValue = opponentValue + abs(opponentLoc[0]-i) + abs(opponentLoc[1]-j)
            return opponentValue-myValue+50*state.cookiecounts[0]-100*state.cookiecounts[1]
        #print state.grid
        elif state.player == 1:
            for i in range(3):
                for j in range(5):
                    if state.grid[i][j] == '1':
                        myLoc[0] = i
                        myLoc[1] = j
                    if state.grid[i][j] == '0':
                        opponentLoc[0] = i
                        opponentLoc[1] = j
            # Manhatten distance
            for i in range(3):
                for j in range(5):
                    if state.grid[i][j] == 'c':
                        myValue = myValue + abs(myLoc[0]-i) + abs(myLoc[1]-j)
                        opponentValue = opponentValue + abs(opponentLoc[0]-i) + abs(opponentLoc[1]-j)
        #print state.grid
            return opponentValue-myValue+50*state.cookiecounts[1]-100*state.cookiecounts[0]
        raise NotImplementedError()

    def minimax_move(self, state):
        assert state.player == self.player_id
        # Experiment with the value of horizon
        horizon = 7
        val, action = minimax_dfs(self.game, state, 0, horizon, self.evaluate)
        print ("Count Minimax: %s" % (countMinimax))
        return action

    def alpha_beta_move(self, state):
        # *** YOUR CODE HERE ***
        horizon = 7
        val, action = alphabeta_dfs(self.game, state, 0, horizon, self.evaluate, -sys.maxint, sys.maxint)
        print ("Count Alpha: %s" % (countAlpha))
        return action
        raise NotImplementedError()
