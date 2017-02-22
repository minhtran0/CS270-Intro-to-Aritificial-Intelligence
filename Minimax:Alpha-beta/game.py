from copy import deepcopy

class PickupState:
    def __init__(self, grid, player=0, cookiecounts=None):
        self.grid = grid
        self.player = player
        self.cookiecounts = cookiecounts
        if cookiecounts == None:
            self.cookiecounts = [0,0]

class PickupGame:
    def __init__(self):
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]
        acts = ['n', 's', 'e', 'w', 'z']
        self.dirs_acts = zip(dirs, acts)
    
    def get_successors(self, state):
        position = None
        grid = state.grid
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == str(state.player):
                    position = (i, j)
                    break
        assert position is not None, "Current player not found in the provided state"
        
        successors = []
        actions = []
        for d, a in self.dirs_acts:
            new_position = (position[0] + d[0], position[1] + d[1])
            if new_position[0] < 0 or new_position[0] >= len(grid) or new_position[1] < 0 or new_position[1] >= len(grid[0]):
                # new position is outside the grid
                continue
            if grid[new_position[0]][new_position[1]] not in ['', 'c']:
                # new position is occupied by the other agent
                continue
            
            new_state = deepcopy(state)
            new_state.player = 1 - state.player
            new_state.grid[position[0]][position[1]] = ''
            new_state.grid[new_position[0]][new_position[1]] = str(state.player)
            if grid[new_position[0]][new_position[1]] == 'c':
                new_state.cookiecounts[state.player] += 1
            
            successors.append(new_state)
            actions.append(a)
            
        return successors, actions

pickup_game = PickupGame()
