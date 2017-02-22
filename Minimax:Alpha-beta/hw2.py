import MalmoPython
import os
import sys
import time
import json
import copy
from game import pickup_game, PickupState
from agents import BasicPlayer, StudentPlayer

num_cookies = 13
actions = {'n':['movenorth 1', 'movenorth 1', 'movenorth 1'],
           's':['movesouth 1', 'movesouth 1', 'movesouth 1'],
           'e':['moveeast 1', 'moveeast 1', 'moveeast 1'],
           'w':['movewest 1', 'movewest 1', 'movewest 1'],
           'z':[]}

def pretty_print_grid(grid):
    print
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print grid[i][j],
        print ""

def extract_observation(msg):
    ob_dict = json.loads(msg)
    ent = ob_dict.get(u'ent')
    count = ob_dict.get(u'Hotbar_0_size', 0)
    grid = [['']*5, ['']*5, ['']*5]

    for e in ent:
        x = int(round((e['x'] - 0.5) / 3))
        z = int(round((e['z'] - 0.5) / 3))
        if e[u'name'] in [u'Agent0', u'Cristina']:
            grid[x][z] = '0'
        elif e[u'name'] == u'Agent1':
            grid[x][z] = '1'
        elif e[u'name'] == u'cookie':
            grid[x][z] = 'c'

    return grid, count


def main(get_agent0_action, get_agent1_action):
    client_pool = MalmoPython.ClientPool()
    client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10000))
    client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10001))

    agent_host0 = MalmoPython.AgentHost()
    agent_host0.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)

    agent_host1 = MalmoPython.AgentHost()
    agent_host1.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)

    mission_file = './hw2.xml'
    my_mission = None
    with open(mission_file, 'r') as f:
        print "Loading mission from %s" % mission_file
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host0.startMission(my_mission, client_pool, MalmoPython.MissionRecordSpec(), 0, '')
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print "Error starting mission:", e
                exit(1)
            else:
                time.sleep(2)

    time.sleep(10)

    max_retries = 30
    for retry in range(max_retries):
        try:
            agent_host1.startMission(my_mission, client_pool, MalmoPython.MissionRecordSpec(), 1, '')
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print "Error starting mission:", e
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print "Waiting for the mission to start ",
    world_state0 = agent_host0.peekWorldState()
    while not world_state0.is_mission_running:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state0 = agent_host0.peekWorldState()
        for error in world_state0.errors:
            print "Error:", error.text

    print
    print "Mission running ",
    print

    cookie_counts = [0, 0]
    while world_state0.is_mission_running:
        time.sleep(2.0)

        world_state0 = agent_host0.getWorldState()
        if world_state0.is_mission_running:
            if cookie_counts[0] + cookie_counts[1] >= num_cookies:
                break

            msg = world_state0.observations[-1].text
            grid, count = extract_observation(msg)

            s = PickupState(grid, 0, cookie_counts)
            action = get_agent0_action(s)
            print 'Agent0 taking action: {0}'.format(action)
            for command in actions[action]:
                agent_host0.sendCommand(command)
                time.sleep(0.1)

            world_state0 = agent_host0.peekWorldState()
            if world_state0.is_mission_running:
                msg = world_state0.observations[-1].text
                grid, count = extract_observation(msg)
                cookie_counts[0] = count

        time.sleep(2.0)

        world_state1 = agent_host1.getWorldState()
        if world_state1.is_mission_running:
            if cookie_counts[0] + cookie_counts[1] >= num_cookies:
                break

            msg = world_state1.observations[-1].text
            grid, count = extract_observation(msg)
            cookie_counts[1] = count

            print cookie_counts
            if cookie_counts[0] + cookie_counts[1] >= num_cookies:
                break

            s = PickupState(grid, 1, cookie_counts)
            action = get_agent1_action(s)
            print 'Agent1 taking action: {0}'.format(action)
            for command in actions[action]:
                agent_host1.sendCommand(command)
                time.sleep(0.1)

            world_state1 = agent_host1.peekWorldState()
            if world_state1.is_mission_running:
                msg = world_state1.observations[-1].text
                grid, count = extract_observation(msg)
                cookie_counts[1] = count

    if cookie_counts[0] > cookie_counts[1]:
        print "Agent0 wins with a score of {0} - {1}".format(cookie_counts[0], cookie_counts[1])
    elif cookie_counts[0] == cookie_counts[1]:
        print "Tie with a score of {0} - {1}".format(cookie_counts[0], cookie_counts[1])
    elif cookie_counts[0] < cookie_counts[1]:
        print "Agent1 wins with a score of {0} - {1}".format(cookie_counts[0], cookie_counts[1])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("agent0", type=str, choices=['basic', 'student'])
    parser.add_argument("agent1", type=str, choices=['basic', 'student'])
    parser.add_argument("agent0alg", type=str, choices=['minimax', 'alphabeta'])
    parser.add_argument("agent1alg", type=str, choices=['minimax', 'alphabeta'])
    args = parser.parse_args()

    if args.agent0 == 'basic':
        agent0 = BasicPlayer(0, pickup_game)
    else:
        agent0 = StudentPlayer(0, pickup_game)

    if args.agent0alg == 'minimax':
        get_agent0_action = agent0.minimax_move
    else:
        get_agent0_action = agent0.alpha_beta_move

    if args.agent1 == 'basic':
        agent1 = BasicPlayer(1, pickup_game)
    else:
        agent1 = StudentPlayer(1, pickup_game)

    if args.agent1alg == 'minimax':
        get_agent1_action = agent1.minimax_move
    else:
        get_agent1_action = agent1.alpha_beta_move

    main(get_agent0_action, get_agent1_action)