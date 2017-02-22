# Note: You do not need to change anything in this file

import MalmoPython
import os
import sys
import time
import json
import copy
from search import breadth_first_search, greedy_search, MazeProblem, pretty_print_grid

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

def GetMissionXML( seed, gp ):
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"/>
                  <DrawingDecorator>
                    <DrawSphere x="-27" y="70" z="0" radius="30" type="air"/>
                  </DrawingDecorator>
                  <MazeDecorator>
                    <Seed>'''+str(seed)+'''</Seed>
                    <SizeAndPosition width="10" length="10" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
                    <StartBlock type="emerald_block" fixedToEdge="true"/>
                    <EndBlock type="redstone_block" fixedToEdge="true"/>
                    <PathBlock type="diamond_block"/>
                    <FloorBlock type="air"/>
                    <GapBlock type="air"/>
                    <GapProbability>'''+str(gp)+'''</GapProbability>
                    <AllowDiagonalMovement>false</AllowDiagonalMovement>
                  </MazeDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="600000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>HW1Bot</Name>
                <AgentStart>
                    <Placement x="0.5" y="56.0" z="0.5"/>
                </AgentStart>
                <AgentHandlers>
                    <DiscreteMovementCommands/>
                    <RewardForTouchingBlockType>
                        <Block type="redstone_block" reward="100.0" />
                    </RewardForTouchingBlockType>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="redstone_block"/>
                    </AgentQuitFromTouchingBlockType>
                    
                    <ObservationFromGrid>
                        <Grid name="front20x10">
                            <min x="-10" y="-1" z="0"/>
                            <max x="10" y="-1" z="10"/>
                        </Grid>
                    </ObservationFromGrid>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

# Create default Malmo objects:
time.sleep(5) # helps recording the video
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
    
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)

if "gs" in sys.argv:
    search_alg = 'gs'
else:
    search_alg = 'bfs'

my_mission = MalmoPython.MissionSpec(GetMissionXML("random", 0.2), True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print "Error starting mission:",e
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print "Waiting for the mission to start ",
world_state = agent_host.getWorldState()
while not world_state.is_mission_running:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print "Error:",error.text

print
print "Mission running ",

commands = {"n": "movenorth 1",
            "s": "movesouth 1",
            "e": "moveeast 1",
            "w": "movewest 1"}

# Loop until mission ends:
while world_state.is_mission_running:
    sys.stdout.write(".")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    
    if world_state.number_of_observations_since_last_state > 0:
        msg = world_state.observations[-1].text
        ob = json.loads(msg)
    
        # get the data in front of the agent ONCE
        all_tiles = ob.get(u'front20x10',0)
        grid = []
        for i in range(10):
            row = (all_tiles[i*21:((i+1)*21)])[::-1] 
            grid.append(row)
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == u'air':
                    grid[i][j] = 'a'
                elif grid[i][j] == u'diamond_block':
                    grid[i][j] =  'd'
                elif grid[i][j] == u'emerald_block':
                    grid[i][j] = 'E'
                elif grid[i][j] == u'redstone_block':
                    grid[i][j] = 'R'
                else:
                    grid[i][j] = '?'
        
        pretty_print_grid(grid)
        
        problem = MazeProblem(grid)
        if search_alg == 'bfs':
            plan = breadth_first_search(problem)
        elif search_alg == 'gs':
            plan = greedy_search(problem)
        if plan:
            for action in plan:
                print 'action: {0}'.format(action)
                command = commands[action]
                agent_host.sendCommand(command)
                time.sleep(0.5)
        
        world_state = agent_host.getWorldState()
        if world_state.is_mission_running or len(world_state.rewards) == 0 or world_state.rewards[-1].getValue() < 100.0:
            print 'Mission failed: did not reach goal state.'
            break
        else:
            print 'Mission accomplished: goal state reached.'
            break
        
    for error in world_state.errors:
        print "Error:",error.text

print
print "Mission ended"
time.sleep(0.5)


    