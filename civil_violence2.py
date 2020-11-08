import numpy as np
import random
from tkinter import *
import time

# ============================================
# Global variables for the model
# ============================================
nation_dimension = 45  # Will be a (nation_dimension,nation_dimension) matrix
vision_agent = 3  # Agent's vision
influence_cop = 3  # Cop's radius of influence
T = 200  # Tres-hold of agent's activation
T_calm = 150 # Treshold needed for the agent to become quiet
percentage_bad_cops=0.1
k = 1  # Constant for P : estimated arrest probability

# ============================================
# Classes and functions (see end of file for simulation)
# ============================================
class agent():
    def __init__(self, nr):
        self.nr = nr  # Identifier
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns initial random position on the matrix
        # locations_agents_dic[nr] = self.position
        self.H = np.random.uniform(0, 1)  # Hardship, see paper
        self.status = 0  # 0: quite, 1:active
        self.Il = np.random.uniform(0, 1)  # Percieved Illegitimacy
        self.G = self.H * self.Il  # Grievance
        self.R = np.random.uniform(0, 1)  # Risk aversion
        self.P = 0  # Estimated arrest probability

    def move(self):
        # Moves the agent if the agent is not in jail
        shift = np.random.randint(-vision_agent+1, vision_agent, (2))
        self.position = self.position + shift  # Move
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not leave the matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not leave the matrix
        # locations_agents_dic[self.nr] = self.position

    def active_near_agents(self, agents):
        # Computes number of near active agents
        near_agents = 0
        for agent in agents:
            if agent.status == 1:
                # Only active agents
                pos = agent.position
                if np.linalg.norm(self.position - pos, ord=np.inf) < vision_agent:
                    # If within vision, count
                    near_agents += 1
        if not self.status == 1:
            # To avoid double counting but always count the agent self, see paper
            near_agents += 1
        return near_agents

    def near_cops(self, cops):
        # Counts cops within vision
        near_cops = 0
        for cop in cops:
            pos = cop.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < vision_agent:
                # If within vision count
                near_cops += 1
        return near_cops

    def updateP(self, agents, cops):
        # Updates estimated arrest probability, see paper
        active_agents_near = self.active_near_agents(agents)
        cops_near = self.near_cops(cops)
        self.P = 1 - np.exp(-k * cops_near / active_agents_near)

    def percieved_agressivity_of_cops(self, cops):
        # Sums over cops agressivity within influence radius
        percieved_agressivity = 0
        for cop in cops:
            pos = cop.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < influence_cop:
                # If within vision count
                percieved_agressivity += cop.agressivity
        return percieved_agressivity

    def updateIl(self, cops):
        self.Il=self.Il*np.exp(self.percieved_agressivity_of_cops(cops))

    def updateG(self):
        # Update net risk, see paper
        self.G = self.Il*self.H

    def update_status(self):
        # Updates the agent's status
        if self.G - self.R*self.P > T:
            # Get's active, see paper
            self.status = 1
        elif self.G - self.R*self.P < T_calm:
            # get quite
            self.status = 0
        # don't change status

    def time_step(self, agent, cops):
        # Comptes one time iteration for the given agent
        self.move()
        self.updateP(agents, cops)
        self.updateIl(cops)
        self.updateG()
        self.update_status()
        return self


class cop():
    def __init__(self, nr):
        self.nr = nr  # Identifier
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns randomly initial position
        if random.random()<percentage_bad_cops :
            self.agressivity = 1
        else:
            self.agressivity = -0.1

    def move(self):
        # Moves the cop withing vision
        shift = np.random.randint(-influence_cop+1, influence_cop, (2))
        self.position = self.position + shift
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not exit matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not exit matrix

    def time_step(self, agents):
        # Compute one time iteration for cops
        self.move()
        return self



# ============================================
# Simulation data --> Do the tuning here
# ============================================
n_agents = 220  # Number of considerate agents
n_cops = 40  # Number of considerate cops

# ============================================
# Simulation computation
# ============================================
agents = [agent(n) for n in range(n_agents)]  # Generate the agents
cops = [cop(n) for n in range(n_cops)]  # Generate the cops

positions = np.zeros((nation_dimension, nation_dimension)) - 1  # Initialisation of the matrix
    # Values of positions are:
    # * -1: no one here
    # * 0: quite agent here
    # * 1: active agent here
    # * 2: cop here
for agent in agents:
    pos = agent.position
    positions[pos[0], pos[1]] = agent.status  # Updates matrix data with agents position and status
for cop in cops:
    pos = cop.position
    positions[pos[0], pos[1]] = 2    # Updates matrix data with cops position


t=300 #time in millisecond between to steps
Pix = 8 #size of a square in pixel

def show(cops,agents):
    interface.delete(ALL)
    for cop in cops:
        pos = cop.position
        interface.create_rectangle(Pix*pos[0],Pix*pos[1],Pix*pos[0]+Pix,Pix*pos[1]+Pix,fill="blue")
    for agent in agents:
        pos = agent.position
        stat= agent.status
        color="green"
        if stat == 1:
            color = "red"
        interface.create_rectangle(Pix*pos[0],Pix*pos[1],Pix*pos[0]+Pix,Pix*pos[1]+Pix,fill=color)

"""
def show(cops,agents):
    interface.delete(ALL)
    for cop in cops:
        pos = cop.position
        a=random.randrange(0,nation_dimension)
        b=random.randrange(0,nation_dimension)
        interface.create_rectangle(Pix*a,Pix*b,Pix*a+Pix,Pix*b+Pix,fill="blue")
 """

def loop (cops,agents,step):
    for m in range(100):
        time.sleep(0.001)
        cops = [cop.time_step(agents) for cop in cops]
        agents = [ag.time_step(agents, cops) for ag in agents]
        show(cops,agents) #update the image 
        print("step "+str(m))
        interface.update()

#Graphic interface with tkinter
window = Tk()
window.title("Civil violence")
interface = Canvas(window,height=nation_dimension*Pix,width= nation_dimension*Pix,bg="white")
interface.pack()
loop(cops,agents,0)
window.mainloop()
