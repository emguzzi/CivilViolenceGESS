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
# Global variables for the model
# ============================================
nation_dimension = 40  # Will be a (nation_dimension,nation_dimension) matrix
p_class_1 = 0.6  # Probability for an agent to be in class 1
vision_agent = 7  # Agent's vision
vision_cop = 7  # Cop's vision
L = 0.82  # Legitimacy, must be element of [0,1], see paper
T = 0.1  # Tres-hold of agent's activation
Jmax = 15  # Maximal Jail time for type 0
factor_Jmax1 = 2  # How many time is Jmax for type 1 bigger than for type 0
k = 2.3  # Constant for P : estimated arrest probability
# locations_agents_dic = {}
# locations_cops_dic = {}
prob_arrest_class_1 = 0.7  # Probability, given an arrest is made, that the arrested agent is


# of type 1

# ============================================
# Classes and functions (see end of file for simulation)
# ============================================

class agent():
    def __init__(self, nr, L, Jmax, p_class_1):
        self.nr = nr  # Identifier
        self.vision_agent = vision_agent
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns initial random position on the matrix
        # locations_agents_dic[nr] = self.position
        self.type = np.random.choice(2, size=1, p=[1 - p_class_1, p_class_1])[0]  # Assigns random type 0 or 1
        self.Jmax = Jmax + self.type * Jmax * (factor_Jmax1 - 1)  # Maximal Jail time for agent (depends on type)
        self.H = np.random.uniform(0, 1)  # Hardship, see paper
        self.status = 0  # 0: quite, 1:active, 2:Jail
        self.G = self.H * (1 - L)  # Grievance, see paper
        self.R = np.random.uniform(0, 1)  # Risk aversion
        self.J = 0  # Remaining jail time, 0 if free
        self.P = 0  # Estimated arrest probability
        self.N = self.R * self.P * self.Jmax  # Agent's net risk

    def move(self):
        # Moves the agent if the agent is not in jail
        shift = np.random.randint(-self.vision_agent, self.vision_agent+1, (2))
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
                if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_agent:
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
            if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_agent:
                # If within vision count
                near_cops += 1
        return near_cops

    def updateP(self, agents, cops):
        # Updates estimated arrest probability, see paper
        active_agents_near = self.active_near_agents(agents)
        cops_near = self.near_cops(cops)
        self.P = 1 - np.exp(-k * cops_near / active_agents_near)

    def updateN(self):
        # Update net risk, see paper
        self.N = self.R * self.P * self.Jmax

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
        self.vision_cop = vision_cop
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns randomly initial position

    def move(self):
        # Moves the cop within vision
        shift = np.random.randint(-self.vision_cop, self.vision_cop+1, (2))
        self.position = self.position + shift
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not exit matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not exit matrix

    # def near_agents(self, agents):
    #     # Compute nume
    #     near_agents = 0
    #     for agent in agents:
    #         if agent.status == 2:
    #             continue
    #         pos = agent.position
    #         if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_cop:
    #             near_agents += 1
    #     return near_agents

    def update_agent_status(self, agents):
        # Arrests randomly (with bias based on type) an agent within vision
        near_active_agents_0 = []  # List type 0 agents within vision
        near_active_agents_1 = []  # List type 1 agents within vision
        for agent in agents:
            pos = agent.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_cop:
                if agent.status == 1:
                    if agent.type == 0:
                        near_active_agents_0.append(agent)
                    else:
                        near_active_agents_1.append(agent)
        if len(near_active_agents_0) > 0:
            if len(near_active_agents_1) > 0:
                # Both types in vision, compute now which type to arrest
                choice01 = np.random.choice(2, 1, p=[1 - prob_arrest_class_1, prob_arrest_class_1])[0]
                if choice01 == 0:
                    # Arrest randomly in type 0
                    random.choice(near_active_agents_0).update_status(arrest=True)
                else:
                    # Arrest randomly in type 1
                    random.choice(near_active_agents_1).update_status(arrest=True)
            else:
                # No type 1 but type 0 in vision, arrest randomly type 0
                random.choice(near_active_agents_0).update_status(arrest=True)
        elif len(near_active_agents_1) > 0:
            # No type 0 vut type 1 in vision, arrest randomly type 1
            random.choice(near_active_agents_1).update_status(arrest=True)

    # No one in vision, no arrest

    def time_step(self, agents):
        # Compute one time iteration for cops
        self.move()
        self.update_agent_status(agents)  # Do arrest if possible
        return self


# ============================================
# Simulation data --> Do the tuning here
# ============================================
n_agents = int(0.7*nation_dimension**2)  # Number of considerate agents
n_cops = int(0.04*nation_dimension**2)  # Number of considerate cops
tfin = 200  # Final time, i.e. number of time steps to consider
agents = [agent(n, L, Jmax, p_class_1) for n in range(n_agents)]  # Generate the agents
cops = [cop(n) for n in range(n_cops)]  # Generate the cops

save = True            # Set to True if want to save the data
interactive = True      # If true computes the html slider stuff

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

color_name_list = ["white", "green", "red", "darkorange", "lime", "fuchsia", "goldenrod", "blue"]

for t in trange(tfin):
    # Does the t-th time iteration
    positions = np.zeros((nation_dimension, nation_dimension)) - 1  # Initialisation of the matrix
    # Values of positions are:
    # * -1: no one here
    # * 0: quite agent type 0 here
    # * 1: active agent type 0 here
    # * 2: agent in jail type 0 here
    # * 3: quite agent type 1 here
    # * 4: active agent type 1 here
    # * 5: agent in jail type 1 here
    # * 6: cop here
    for agent in agents:
        pos = agent.position
        positions[pos[0], pos[1]] = agent.status + 3*agent.type  # Updates matrix data with agents position and status
    for cop in cops:
        pos = cop.position
        positions[pos[0], pos[1]] = 6                   # Updates matrix data with cops position
    positions_data[t, :, :] = positions         # Stores the data of the positons

    im = plt.imshow(positions, cmap=mpl.colors.ListedColormap(color_name_list))
    values = [-1, 0, 1, 2, 3, 4, 5, 6]
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    # Plots the positions matrix
    if save:
        plt.savefig(name_to_save + '_time_iter_nr' + str(t) + '.png')
        # Saves the positions matrix
    # Compute now one time steps for each cop and each agent
    cops = [cop.time_step(agents) for cop in cops]
    agents = [ag.time_step(agents, cops) for ag in agents]


if interactive:
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in np.arange(0, tfin, 1):
        curent_pd = pd.DataFrame(positions_data[step, :, :])
        print(type(curent_pd.applymap(str)[0][0]))
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
             'p_class_1' + ': ' + str(p_class_1),
             'vision_agent' + ': ' + str(vision_agent),
             'vision_cop' + ': ' + str(vision_cop),
             'L' + ': ' + str(L),
             'T' + ': ' + str(T),
             'Jmax' + ': ' + str(Jmax),
             'factor_Jmax1' + ': ' + str(factor_Jmax1),
             'k' + ': ' + str(k),
             'prob_arrest_class_1' + ': ' + str(prob_arrest_class_1),
             'n_agents' + ': ' + str(n_agents),
             'n_cops' + ': ' + str(n_cops),
             'tfin' + ': ' + str(tfin)]

    with open(name_to_save+'_par.txt','w') as file:
        for line in lines:
            file.write(line + '\n')
        file.close()
