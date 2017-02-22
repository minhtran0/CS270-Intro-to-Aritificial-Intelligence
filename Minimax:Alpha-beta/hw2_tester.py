"""
Created on February 15, 2017

@author: rhondusmithwick
"""
from copy import deepcopy
from game import pickup_game, PickupState
from agents import BasicPlayer, StudentPlayer
from numpy import argmax

ACTION_DIRS = {'n': (0, -1), 's': (0, 1), 'e': (1, 0),
               'w': (-1, 0), 'z': (0, 0)}
COOKIE_ENTITY = 'c'
NO_ENTITY = ''


def pretty_print_grid(grid):
    print
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print grid[i][j],
        print ""


def is_terminal(grid):
    """
    Determines if this grid is terminal, that is, it has no more cookies.
    :param grid: the grid to test
    :return: true if the grid is terminal
    """
    return len(get_entity_locations(grid, COOKIE_ENTITY)) == 0


def get_entity_locations(grid, entity):
    """
    Returns a set of the locations of this entity in the grid.
    :param grid: the grid
    :param entity: the entity (IE '1', '0', 'c')
    :return: a set of location tuples where this entity is in the grid
    """
    grid_locations = set()
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == entity:
                grid_locations.add((i, j))
    return grid_locations


def make_move(grid, cookie_counts, player, get_action):
    """
    Execute a move with the given arguments.
    :param grid: The grid to work with
    :param cookie_counts: The current cookie counts
    :param player: The player who is making a move
    :param get_action: The action for the player
    :return: a tuple of the new grid and cookie counts.
    """
    print "player_id Moving: " + str(player)
    working_grid = deepcopy(grid)
    working_cookie_counts = deepcopy(cookie_counts)
    state = PickupState(working_grid, player, working_cookie_counts)
    action = get_action(state)
    print 'Agent{} taking action: {}'.format(player, action)
    old_position = get_entity_locations(working_grid, str(state.player)).pop()
    direction = ACTION_DIRS[action]
    new_position = (old_position[0] + direction[0], old_position[1] + direction[1])
    if working_grid[new_position[0]][new_position[1]] == COOKIE_ENTITY:
        working_cookie_counts[state.player] += 1
    working_grid[old_position[0]][old_position[1]] = NO_ENTITY
    working_grid[new_position[0]][new_position[1]] = str(state.player)
    return working_grid, working_cookie_counts


def cookie_game(grid, get_agent0_action, get_agent1_action):
    """
    Plays the game.
    :param grid: the starting grid
    :param get_agent0_action: the method to get an action for agent 0
    :param get_agent1_action: the method to get an action for agent 1
    """
    cookie_counts = [0, 0]
    while True:
        grid, cookie_counts = make_move(grid, cookie_counts, 0, get_agent0_action)
        if is_terminal(grid):
            break
        grid, cookie_counts = make_move(grid, cookie_counts, 1, get_agent1_action)
        if is_terminal(grid):
            break
    winner = argmax(cookie_counts)
    print "Agent{} wins with a score of {} - {}".format(winner, cookie_counts[0], cookie_counts[1])


if __name__ == "__main__":
    GRID = ["c 0 c c c".split(),
            "c c c c c".split(),
            "c c 1 c c".split()]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("agent0", type=str, choices=['basic', 'student'])
    parser.add_argument("agent1", type=str, choices=['basic', 'student'])
    parser.add_argument("agent0alg", type=str, choices=['minimax', 'alphabeta'])
    parser.add_argument("agent1alg", type=str, choices=['minimax', 'alphabeta'])
    args = parser.parse_args()

    agent0 = BasicPlayer(0, pickup_game) if args.agent0 == 'basic' else StudentPlayer(0, pickup_game)
    get_agent0_action = agent0.minimax_move if args.agent0alg == 'minimax' else agent0.alpha_beta_move

    agent1 = BasicPlayer(1, pickup_game) if args.agent1 == 'basic' else StudentPlayer(1, pickup_game)
    get_agent1_action = agent1.minimax_move if args.agent1alg == 'minimax' else agent1.alpha_beta_move
    cookie_game(GRID, get_agent0_action, get_agent1_action)
