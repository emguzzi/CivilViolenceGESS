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

model = 2

nation_dimension = 40   # defines size of matrix (nation_dimension * nation_dimension)
vision = 5              # vision of each agent / cop
L = 0.82                # Legitimacy
T = 0.15                # Threshold for agent's activation
Jmax =  15              # max jail time
k = 2.3                 # constant for making P reasonable

if model == 1:
    D_const = [1,2]                 # constant for calculating discrimination
    prob_arrest_class_1 = 0.7       # probability for type 1 to be arrested
    factor_Jmax1 = 2                # factor for the max jail time of type 1
    p_class_1 = 0.6                 # percentage of agents for type 1
    percentage_bad_cops = 0         # percentage of bad cops
    aggressivity_bad_cops = 0.0
    aggressivity_good_cops = 0.0
elif model == 2:
    D_const = [0,0]
    prob_arrest_class_1 = 0
    factor_Jmax1 = 1
    p_class_1 = 0
    percentage_bad_cops = 0.02
    aggressivity_bad_cops = -1
    aggressivity_good_cops = -1

# ============================================
# Simulation data --> Do the tuning here
# ============================================
p_agents = 0.7  # Number of considerate agents
p_cops = 0.04  # Number of considerate cops
tfin = 80 # Final time, i.e. number of time steps to consider

save = True             # Set to True if want to save the data
interactive = True      # If true computes the html slider stuff
show_plot = False

class Agent():
    # Initialization of new agent. For the meaning of the variables, check the paper
    def __init__(self, position, L, Jmax, p_class_1):
        self.position = position
        self.type = np.random.choice(2, size=1, p=[1 - p_class_1, p_class_1])[0]
        self.Jmax = Jmax + self.type * Jmax * (factor_Jmax1 - 1)
        self.H = np.random.uniform(0, 1)
        self.status = 0
        self.G = self.H * (1 - L)
        self.R = np.random.uniform(0, 1)
        self.J = 0
        self.P = 0
        self.N = self.R * self.P
        self.D = 0
        self.Il = 1-L

    # updates status -> active / not active / jail
    def update_status(self):
        if self.status == 2 and self.J > 0:
            self.J = self.J - 1
        elif self.status == 2 and self.J == 0:
            self.status = 1
        elif self.G - self.N > T:
            self.status = 1
        else:
            self.status = 0

    # calculate near active agents
    def active_near_agents(self):
        near_agents = 0
        all_near = get_nearby_agents(self.position[0], self.position[1], vision)
        for agnt in all_near:
            if isinstance(agnt, Agent):
               near_agents += 1
        return near_agents

    # find near cops
    def near_cops(self):
        near_cops = 0
        all_near = get_nearby_agents(self.position[0], self.position[1], vision)
        for agent in all_near:
            if isinstance(agent, Cop):
                near_cops += 1
        return near_cops

    # calculate arrest probability
    def updateP(self):
        active_agents_near = self.active_near_agents()
        cops_near = self.near_cops()
        self.P = 1 - np.exp(-k * cops_near / max(1.0,active_agents_near))

    # calculate overall aggressivity of near cops
    def percieved_aggressivity_of_cops(self):
        percieved_aggressivity = 0
        for cop in [cop for cop in get_nearby_agents(self.position[0], self.position[1], vision) if isinstance(cop, Cop)]:
            percieved_aggressivity += cop.aggressivity
        return percieved_aggressivity

    # move to new empty field around the current field
    def move(self):
        possible_positions = get_empty_field(self.position[0], self.position[1], vision)
        old_position = self.position
        if possible_positions:
            new_position = random.choice(possible_positions)
            positions[new_position[0]][new_position[1]] = positions[old_position[0]][old_position[1]]
            positions[old_position[0]][old_position[1]] = None
            self.position = new_position

    # calculate the arrested ratio
    def compute_arrested_ratio(self, range):
        tot_arrested = 0
        type_1_arrested = 0
        for agent in [agent for agent in get_nearby_agents(self.position[0], self.position[1], range) if isinstance(agent, Agent)]:
            if agent.status == 2:
                tot_arrested += 1
                if agent.type == 1:
                    type_1_arrested += 1
        if tot_arrested == 0:
            return 0.5
        else:
            return type_1_arrested / tot_arrested

    def updateIl(self):
        self.Il=self.Il*np.exp(self.percieved_aggressivity_of_cops())

    def updateN(self):
        self.N = self.R * self.P

    def updateG(self):
        self.G = self.Il*self.H

    def updateD(self):
        radius = 40
        ratio = self.compute_arrested_ratio(radius)
        self.D = D_const[self.type]*abs(0.5 - ratio)

    def arrest(self):
        self.J = np.random.randint(1, Jmax)
        self.status = 2

    def time_step(self):
        if self.status != 2:
            self.updateP()
            self.updateN()
            self.updateG()
            self.updateD()
            self.move()
            if model == 2:
                self.updateIl()
        self.update_status()
        return self


class Cop():
    # initialize new element of cop
    def __init__(self, position):
        self.position = position
        self.aggressivity = random.choices([aggressivity_bad_cops, aggressivity_good_cops],[percentage_bad_cops, 1-percentage_bad_cops])[0]

    # move to empty field nearby
    def move(self):
        possible_positions = get_empty_field(self.position[0], self.position[1], vision)
        old_position = self.position
        if possible_positions:
            new_position = random.choice(possible_positions)
            positions[new_position[0]][new_position[1]] = positions[old_position[0]][old_position[1]]
            positions[old_position[0]][old_position[1]] = None
            self.position = new_position

    # arrest an active agent
    def update_agent_status(self):
        near_active_agents_0 = []  # List type 0 agents within vision
        near_active_agents_1 = []  # List type 1 agents within vision
        nearby_agents_and_cops = get_nearby_agents(self.position[0], self.position[1], vision)
        nearby_agents = [agnt for agnt in nearby_agents_and_cops if isinstance(agnt, Agent)]

        for agnt in nearby_agents:
            if agnt.status == 1:
                if agnt.type == 0:
                    near_active_agents_0.append(agnt)
                else:
                    near_active_agents_1.append(agnt)

        if len(near_active_agents_0) > 0:
            if len(near_active_agents_1) > 0:
                choice01 = np.random.choice(2, 1, p=[1 - prob_arrest_class_1, prob_arrest_class_1])[0]
                if choice01 == 0:
                    # Arrest randomly in type 0
                    random.choice(near_active_agents_0).arrest()
                else:
                    # Arrest randomly in type 1
                    random.choice(near_active_agents_1).arrest()
            else:
                # No type 1 but type 0 in vision, arrest randomly type 0
                random.choice(near_active_agents_0).arrest()
        elif len(near_active_agents_1) > 0:
            # No type 0 vut type 1 in vision, arrest randomly type 1
            random.choice(near_active_agents_1).arrest()

    def time_step(self):
        self.move()
        self.update_agent_status()
        return self

# ============================================
# Simulation computation
# ============================================

now = datetime.now()    # Gets date and time info
dt_string = now.strftime("%d_%m_%Y_%H_%M")
name_to_save = 'simulation_' + dt_string
if save:
    if not os.path.isdir(name_to_save):
        # If save and directory does not exists, create one
        os.mkdir(name_to_save)
name_to_save = name_to_save + '/' + name_to_save

agents = []
cops = []
positions = []
for i in range(nation_dimension):
    line = []
    for j in range(nation_dimension):
        rand = random.random()
        if rand < p_agents:
            agent_instance = Agent([i, j], L, Jmax, p_class_1)
            line.append(agent_instance)
            agents.append(agent_instance)
        elif rand < (p_agents + p_cops):
            cop_instance = Cop([i, j])
            line.append(cop_instance)
            cops.append(cop_instance)
        else:
            line.append(None)
    positions.append(line)

def get_nearby_agents(x, y, local_vision):
    nearby_agents = []
    for i in range(max(x - local_vision, 0), min(x + local_vision, nation_dimension)):
        for j in range(max(y - local_vision, 0), min(y + local_vision, nation_dimension)):
            if i is not x or j is not y:
                if positions[i][j] is not None:
                    nearby_agents.append(positions[i][j])
    return nearby_agents

def get_empty_field(x,y,distance):
    empty_fields = []
    for i in range(max(x-distance, 0), min(x+1+distance, nation_dimension)):
        for j in range(max(y-distance, 0), min(y+1+distance, nation_dimension)):
            if positions[i][j] is None:
                    empty_fields.append([i,j])
    return empty_fields
 

color_name_list = ["white", "green", "red", "darkorange", "lime", "fuchsia", "goldenrod", "blue", "darkblue"]
time = range(tfin)
D_list = [0]*tfin
arrested_list = [0]*tfin
type_1_arrested_list = [0]*tfin
type_0_arrested_list = [0]*tfin
active_list = [0]*tfin
type_1_active_list = [0]*tfin
type_0_active_list = [0]*tfin
positions_data=np.empty([tfin, nation_dimension, nation_dimension])
for t in trange(tfin):
    arrested = 0
    type_1_arrested = 0
    type_0_arrested = 0
    active = 0
    type_1_active = 0
    type_0_active = 0
    D = 0
    # Does the t-th time iteration
    current_status = [] 
    # Values of positions are:
    # * -1: no one here
    # * 0: quite agent type 0 here
    # * 1: active agent type 0 here
    # * 2: agent in jail type 0 here
    # * 3: quite agent type 1 here
    # * 4: active agent type 1 here
    # * 5: agent in jail type 1 here
    # * 6: cop here
   
    for i in range(nation_dimension):
        line = []
        for j in range(nation_dimension):
            element = positions[i][j]
            if element is None:
               line.append(-1)
            elif isinstance(element, Cop):
               if element.aggressivity < 0:
                   line.append(6)
               else:
                   line.append(7)
            elif isinstance(element, Agent):
               line.append(element.status + 3*element.type)
               if element.status == 2:
                   arrested = arrested + 1
                   if element.type == 1:
                       type_1_arrested = type_1_arrested + 1
                   else:
                       type_0_arrested = type_0_arrested + 1
               elif element.status == 1:
                   active = active + 1
                   if element.type == 1:
                       type_1_active = type_1_active + 1
                   else:
                       type_0_active = type_0_active + 1
        current_status.append(line)
    positions_data[t, :, :] = current_status       # Stores the data of the positons
    im = plt.imshow(current_status, cmap=mpl.colors.ListedColormap(color_name_list, N=5))
    values = [-1, 0, 1, 2, 3, 4, 5, 6]
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if show_plot:
        plt.show()
    # Plots the positions matrix
    if save:
        plt.savefig(name_to_save + '_time_iter_nr' + str(t) + '.png')
        # Saves the positions matrix
    # Compute now one time steps for each cop and each agent
    for cop in cops:
        cop.time_step()
    for ag in agents:
        ag.time_step()
    D_list[t] = D
    arrested_list[t] = arrested
    type_1_arrested_list[t] = type_1_arrested
    type_0_arrested_list[t] = type_0_arrested
    active_list[t] = active
    type_1_active_list[t] = type_1_active
    type_0_active_list[t] = type_0_active

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
        active=60,
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
             'vision' + ': ' + str(vision),
             'L' + ': ' + str(L),
             'T' + ': ' + str(T),
             'Jmax' + ': ' + str(Jmax),
             'factor_Jmax1' + ': ' + str(factor_Jmax1),
             'k' + ': ' + str(k),
             'prob_arrest_class_1' + ': ' + str(prob_arrest_class_1),
             'p_agents' + ': ' + str(p_agents),
             'p_cops' + ': ' + str(p_cops),
             'tfin' + ': ' + str(tfin)]

    with open(name_to_save + '_par.txt', 'w') as file:
        for line in lines:
            file.write(line + '\n')
        file.close()
    # plot graphics for type0/1 active/arrested and D
    fig, ax = plt.subplots()
    ax.plot(time, D_list)
    ax.set(xlabel='time (epochs)', ylabel="agent's percieved D", title='Discrimination factor')
    ax.grid()
    ax.legend(['D factor'])
    fig.savefig(name_to_save + 'Discrimination.png')

    fig, ax = plt.subplots()
    ax.plot(time, arrested_list, label='Total number of arrested agents')
    ax.plot(time, type_1_arrested_list, label='Total number of type 1 arrested agents')
    ax.plot(time, type_0_arrested_list, label='Total number of type 0 arrested agents')
    ax.set(xlabel='time (epochs)', ylabel="number of agents", title='Arrested agents')
    ax.grid()
    ax.legend(['total arrested','type 1 arrested', 'type 0 arrested'])
    fig.savefig(name_to_save + 'Arrests.png')

    fig, ax = plt.subplots()
    ax.plot(time, active_list, label='Total number of active agents')
    ax.plot(time, type_1_active_list, label='Total number of type 1 active agents')
    ax.plot(time, type_0_active_list, label='Total number of type 0 active agents')
    ax.set(xlabel='time (epochs)', ylabel="number of agents", title='Active agents')
    ax.grid()
    ax.legend(['total active', 'type 1 active', 'type 0 active'])
    fig.savefig(name_to_save + 'Active.png')
