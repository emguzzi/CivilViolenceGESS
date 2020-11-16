import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from datetime import datetime
import os
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm, trange
# ============================================
# Global variables for the model --> Do the tuning here
# ============================================
nation_dimension = 40  # Will be a (nation_dimension,nation_dimension) matrix
vision_agent = 5  # Agent's vision
vision_cop = 3  # Cop's vision
L = 0.3  # Legitimacy, must be element of [0,1], see paper
T = 0.03  # Tres-hold of agent's activation
Jmax = 9  # Maximal Jail time for type 0
alpha=1.3 # how the jmax term affects the estimated risk to rebel
k = 0.5  # Constant for P : estimated arrest probability
percentage_agent=0.7  #percentage of all the cells occupied by agents
percentage_cops=0.037 #percentage of all the cells occupied by cops
tfin = 150 # Final time, i.e. number of time steps to consider
# ============================================
# Classes and functions (see end of file for simulation)
# ============================================

class agent():
    def __init__(self, nr):
        self.nr = nr  # Identifier
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns initial random position on the matrix
        # locations_agents_dic[nr] = self.position
        self.H = np.random.uniform(0, 1)  # Hardship, see paper
        self.status = 0  # 0: quite, 1:active, 2:Jail
        self.G = self.H * (1 - L)  # Grievance, see paper
        self.R = np.random.uniform(0, 1)  # Risk aversion
        self.J = 0  # Remaining jail time, 0 if free
        self.P = 0  # Estimated arrest probability
        self.N = self.R * self.P * Jmax  # Agent's net risk
     

    def move(self):
        # Moves the agent if the agent is not in jail
        shift = np.random.randint(-1, 2, (2))
        if self.status == 2:  # Check for status
            shift = np.array([0, 0])
        self.position = self.position + shift  # Move
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not leave the matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not leave the matrix
        # locations_agents_dic[self.nr] = self.position

    def update_status(self, arrest=False):
        # Updates the agent's status
        if self.status == 2 and self.J > 0:
            # If in Jail, Jail time reduces by 1
            self.J = self.J - 1

        elif self.status == 2 and self.J == 0:
            # Exits Jail and is now active
            self.status = 1
        elif arrest:
            # Is arrested and assigned a Jail time, see paper
            self.status = 2
            self.J = np.random.randint(1, Jmax)
        elif self.G - self.N > T:
            # Get's active, see paper
            self.status = 1
            """print("G:",self.G)
            print(" N:",self.N)
            print(" P:",self.P)
            print(" R:", self.R)"""
        else:
            # Keep quite
            self.status = 0

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

    def updateP(self, agent, cop):
        # Updates estimated arrest probability, see paper
        active_agents_near = self.active_near_agents(agent)
        cops_near = self.near_cops(cop)
        self.P = 1 - np.exp(-k * (cops_near+1) / (active_agents_near)) 
        # +1 for cops near such that it doesn't output 0 when there is no cop in vision

    def updateN(self):
        # Update net risk, see paper
        self.N = self.R * self.P * (Jmax**alpha)


    def time_step(self, agent, cops):
        # Comptes one time iteration for the given agent
        self.move()
        self.updateP(agent, cops)
        self.updateN()
        self.update_status(arrest=False)
        return self


class cop():
    def __init__(self, nr):
        self.nr = nr  # Identifier
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns randomly initial position

    def move(self):
        # Moves the cop within vision
        shift = np.random.randint(-1, 2, (2))
        self.position = self.position + shift
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not exit matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not exit matrix

    def update_agent_status(self, agents):
        # Arrests randomly (with bias based on type) an agent within vision
        near_active_agents = []  # List agents within vision
        for agent in agents:
            pos = agent.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < vision_cop:
                if agent.status == 1:
                    near_active_agents.append(agent)

        if len(near_active_agents) > 0:
            random.choice(near_active_agents).update_status(arrest=True)
        # No one activ in vision, no arrest

    def time_step(self, agents):
        # Compute one time iteration for cops
        self.move()
        self.update_agent_status(agents)  # Do arrest if possible
        return self


# ============================================
# Simulation data 
# ============================================
n_agents = int(percentage_agent*nation_dimension**2)  # Number of considerate agents
n_cops = int(percentage_cops*nation_dimension**2)  # Number of considerate cops

agents = [agent(n) for n in range(n_agents)]  # Generate the agents
cops = [cop(n) for n in range(n_cops)]  # Generate the cops

save = True            # Set to True if want to save the data
interactive = True      # If true computes the html slider stuff
show_plot = False

# ============================================
# Simulation computation
# ============================================

now = datetime.now()    # Gets date and time info
dt_string = now.strftime("%d_%m_%Y_%H_%M")
name_to_save = 'simulation_' + dt_string    # All will be save with this name + counter + extensions
if save:
    if not os.path.isdir(name_to_save):
        # If save and directory does not exists, create one
        os.mkdir(name_to_save)
name_to_save = name_to_save + '/' + name_to_save

positions_data = np.empty([tfin, nation_dimension, nation_dimension])  # Stores positional and type data


color_name_list = ["white", "green", "red", "black", "blue"]
values = [-1, 0, 1, 2, 3]
names=["empty","quiet","rebel","in jail","police"]

time =range(tfin)
D_list = [0]*len(range(tfin))
arrested_list = [0]*len(range(tfin))
active_list = [0]*len(range(tfin))

for t in trange(tfin):
    arrested = 0
    active = 0
    # Does the t-th time iteration
    positions = np.zeros((nation_dimension, nation_dimension)) - 1  # Initialisation of the matrix
    # Values of positions are:
    # * -1: no one here
    # * 0: quite agent type here
    # * 1: active agent type here
    # * 2: agent in jail here
    # * 3: cop here
    for agent in agents:
        pos = agent.position
        positions[pos[0], pos[1]] = agent.status
        if agent.status == 2:
            arrested = arrested+1
        elif agent.status ==1:
            active = active +1
    for cop in cops:
        pos = cop.position
        positions[pos[0], pos[1]] = 3                   # Updates matrix data with cops position
    positions_data[t, :, :] = positions         # Stores the data of the positons
    im = plt.imshow(positions, cmap=mpl.colors.ListedColormap(color_name_list))
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if show_plot:
        plt.show()
    # Plots the positions matrix
    if save:
        plt.savefig(name_to_save + '_time_iter_nr' + str(t) + '.png')
        # Saves the positions matrix
    # Compute now one time steps for each cop and each agent
    cops = [cop.time_step(agents) for cop in cops]
    agents = [ag.time_step(agents, cops) for ag in agents]
    arrested_list[t] = arrested
    active_list[t] = active

if interactive:

    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in np.arange(0, tfin, 1):
        curent_pd = pd.DataFrame(positions_data[step, :, :])
        fig.add_trace(go.Heatmap(
                z=curent_pd.applymap(str),
            colorscale=color_name_list)
        )
    # Make First trace visible
    fig.data[0].visible = True
    # Create and add slider
    steps = []

    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Time step: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.show()
    if save:
        fig.write_html(name_to_save+'.html')

if save:
    lines = ['nation_dimension' + ': ' + str(nation_dimension),
        
             'vision_agent' + ': ' + str(vision_agent),
             'vision_cop' + ': ' + str(vision_cop),
             'L' + ': ' + str(L),
             'T' + ': ' + str(T),
             'Jmax' + ': ' + str(Jmax),
             'k' + ': ' + str(k),
             'n_agents' + ': ' + str(n_agents),
             'percentage_agent' + ': ' + str(percentage_agent),
             'percentage_cops' + ': ' + str(percentage_cops),
             'n_cops' + ': ' + str(n_cops),
             'tfin' + ': ' + str(tfin)]

    with open(name_to_save+'_par.txt','w') as file:
        for line in lines:
            file.write(line + '\n')
        file.close()

    fig, ax = plt.subplots()
    ax.plot(time, arrested_list,label = 'Total number of arrested agents')
    ax.set(xlabel='time (epochs)', ylabel="number of agents in jail",title='Arrested agents')
    ax.grid()
    fig.savefig(name_to_save+'Arrests.png')

    fig, ax = plt.subplots()
    ax.plot(time, active_list,label = 'Total number of active agents')
    ax.set(xlabel='time (epochs)', ylabel="number of active agents",title='Active agents')
    ax.grid()
    fig.savefig(name_to_save+'Active.png')