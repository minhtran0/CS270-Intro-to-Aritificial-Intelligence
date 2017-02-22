import os
import sys
import time
import copy
from search import breadth_first_search, greedy_search, MazeProblem, pretty_print_grid

curr_marker = 'X'
visited_marker = 'V'

grid = [
['a', 'a','a','a','a','a','a', 'd'],
['a', 'a','a','a','a','a','a', 'd'],
['a', 'd','d','d','d','d','d', 'd'],
['a', 'd','a','a','d','a','d', 'd'],
['a', 'd','d','d','a','R','d', 'd'],
['a', 'd','d','d','d','a','a', 'd'],
['a', 'd','d','d','d','d','a', 'd'],
['a', 'E','d','a','a','d','d', 'd'],
['a', 'a','a','a','a','a','d', 'd'],
['a', 'a','a','a','a','a','a', 'd']
]

actions = {'n': (-1,0), 's': (1,0), 'e': (0,-1), 'w': (0,1)}

def simulate_plan(plan, problem, grid):
    display_grid = copy.deepcopy(grid)
    i, j = problem.get_start_state()
    display_grid[i][j] = curr_marker

    print "-------------------------------------Starting grid"
    pretty_print_grid(display_grid)

    step = 0
    for action in plan:
        time.sleep(1.5)
        step += 1
        display_grid[i][j] = visited_marker
        if action not in actions:
            print "\nAction not defined: " + action
            return
        else:
            d = actions[action]
            i += d[0]
            j += d[1]
            print 'new pos: {0}, {1}'.format(i, j)
            display_grid[i][j] = curr_marker
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 'a':
                print "\nInvalid location visited with action " + action
                return
            if grid[i][j] == 'R':
                print "-------------------------------------Step " + str(step) + ", Action: " + action
                pretty_print_grid(display_grid)
                print "\nSuccess!"
                return
            print "-------------------------------------Step " + str(step) + ", Action: " + action
            pretty_print_grid(display_grid)

if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('search_alg', nargs='?', type=str, choices=['gs', 'bfs'], default='bfs')
    args = parser.parse_args()
    
    problem = MazeProblem(grid)
    if args.search_alg == 'bfs':
        plan = breadth_first_search(problem)
    elif args.search_alg == 'gs':
        plan = greedy_search(problem)
    
    if plan:
        print "\nBeginning test script. '" + curr_marker + "' represents the agent's current grid location while '" + visited_marker + "' represents previously visited grid locations"
        print
        time.sleep(4)
        simulate_plan(plan, problem, grid)
    
    print "\nTest script ended"
    time.sleep(1)
    
    