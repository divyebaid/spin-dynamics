# %%
# -*- coding: utf-8 -*-

"""
This Python script simulates and visualizes the time-dependent evolution of 4 quantum dots potentials landscape affected by a pulsed perturbation. 
It uses a custom double-barrier potential structure with 4 quantum dots, and displays how the potential changes over time when an external impulsion modifies one of the barriers.
"""

# Importing necessary libraries
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import itertools
import scipy.linalg
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import *
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import root_scalar


# Functions:
# Hopping, Potential and Hubbard over time functions:
def get_hopping_simple_matrix_over_time(N, t, time_array):
    """
    Returns a simple hopping matrix for a 1D lattice with N sites.
    The matrix is tridiagonal with t on the off-diagonal elements.

    Parameters:
    N: int, number of sites
    t: float, hopping integral
    time_array: array, time points

    Returns:
    list of arrays, hopping matrices (N×N) for each time point
    """
    t_matrix_2 = []
    for time in time_array:
        t_matrix = np.zeros((N, N))
        for i in range(N-1):
            t_matrix[i, i+1] = t
            t_matrix[i+1, i] = t
        t_matrix_2.append(t_matrix)
    return t_matrix_2

def get_U_matrix_over_time(N, U_value, time_array):
    """
    Returns U matrices (NxN) representing on-site interactions for each site.
    
    Parameters:
    N: int, number of sites
    U_value: float, on-site interaction strength
    time_array: array, time points

    Returns:
    list of arrays, U matrices (NxN) for each time point
    """
    U_matrix_time = []
    for time in time_array:
        U_matrix = np.zeros((N, N))
        # Pour le modèle de Hubbard, seuls les éléments diagonaux sont utilisés
        # U_matrix[i,i] = force d'interaction sur le site i
        for i in range(N):
            U_matrix[i, i] = U_value
        U_matrix_time.append(U_matrix)
    return U_matrix_time

def hubbard_hamiltonian_matrix_modified(N, t, U_matrix, V=0, states=None):
    """
    Returns the Hubbard Hamiltonian matrix for a system of N sites.
    
    Parameters:
    N: int, number of sites
    t: array, hopping integral matrix (NxN matrix)
    U_matrix: array, on-site interaction matrix (NxN matrix)
    V: float, nearest-neighbor interaction strength (default is 0)
    states: array, optional, list of states to consider

    Returns:
    array, Hubbard Hamiltonian matrix in the basis of all possible states (dimxdim)
    """
    
    if states is None:
        states = get_hubbard_states(N)
        dim = len(states)
    else:
        dim = len(states)

    H = np.zeros((dim, dim))
    
    #print(f"Debug: N={N}, dim={dim}, U_matrix.shape={U_matrix.shape}, t.shape={t.shape}")
    
    # Loop over all quantum states (rows)
    for i in range(dim):
        state_i = states[i]
        
        # Loop over all quantum states (columns)
        for j in range(dim):
            state_j = states[j]
            
            # Diagonal elements: Coulomb interaction term 
            if i == j:
                for site in range(N):
                    # Check if both up and down spins are present at the site
                    n_up = number_operator(state_i, site, 'u')
                    n_down = number_operator(state_i, site, 'd')
                    
                    # Utiliser U_matrix[site, site] pour l'interaction sur ce site
                    H[i, j] += U_matrix[site, site] * n_up * n_down
                
                # Nearest-neighbor interaction V
                if V != 0:
                    for site1 in range(N-1):
                        site2 = site1 + 1
                        n1 = number_operator(state_i, site1, 'u') + number_operator(state_i, site1, 'd')
                        n2 = number_operator(state_i, site2, 'u') + number_operator(state_i, site2, 'd')
                        H[i, i] += V * n1 * n2
                
            # Off-diagonal: Hopping terms
            else:
                # Determine if states i and j differ by a single hopping event
                for site1 in range(N):
                    # Hubbard nearest-neighbor hopping
                    for site2 in (site1-1, site1+1):
                        if 0 <= site2 < N:
                            for spin in ['u', 'd']:
                                temp = annihilation(state_i, site1, spin)
                                # Check if there is a spin to move at site1 with spin
                                if np.any(temp):
                                    final = creation(temp, site2, spin)  # 0 if already occupied
                                
                                    if np.array_equal(np.abs(final), state_j):
                                        sign = hopping_term_sign_factor(state_i, site1, site2, spin)
                                        H[i, j] -= t[site1, site2] * sign

    return H

# potential+pulse over time graph :
def pulse_U(t_array, t_start=5, delta_t=1, delta_U=5):
    """
    Génère une impulsion rectangulaire
    
    Paramètres:
    - t : array des temps (ou temps unique)
    - t_start : temps de début de l'impulsion (défaut: 0)
    - delta_t : largeur de l'impulsion (défaut: 1)
    - amplitude : amplitude de l'impulsion (défaut: 1)
    
    Retourne:
    - Valeur(s) de l'impulsion au(x) temps t
    """
    return delta_U * ((t_array >= t_start) & (t_array < t_start + delta_t))

def potential_over_time(U_imp, x_vals,
    dot_positions,
    well_depth=30,
    well_width=10,
    barrier_12=20, barrier_23=80, barrier_34=20,
    width_12=5, width_23=5, width_34=5):
    res=[]
    for imp in U_imp:
        res.append(create_4dots_potential_with_custom_barriers(x_vals,dot_positions,well_depth=well_depth,well_width=well_width,barrier_12=barrier_12, barrier_23=barrier_23-imp, barrier_34=barrier_34,width_12=width_12, width_23=width_23, width_34=width_34))
    return res

def create_4dots_potential_with_custom_barriers(
    x_vals,
    dot_positions,
    well_depth=30,
    well_width=10,
    barrier_12=20, barrier_23=80, barrier_34=20,
    width_12=5, width_23=5, width_34=5
):
    """
    Crée un potentiel avec 4 puits quantiques (gaussiens) et des barrières personnalisées entre eux.

    Parameters:
    - x_vals: grille spatiale (en mètres)
    - dot_positions: positions des puits quantiques (en mètres)
    - well_depth: profondeur des puits (en meV)
    - well_width: largeur des puits (en nm)
    - barrier_ij: hauteur de la barrière entre dot_i et dot_j (en meV)
    - width_ij: largeur de la barrière correspondante (en nm)

    Returns:
    - V_total: potentiel complet (puits + barrières)
    """
    V_total = np.zeros_like(x_vals)
    well_width_m = well_width * 1e-9

    # Ajouter les puits quantiques
    for pos in dot_positions:
        well = -well_depth * np.exp(-((x_vals - pos)**2) / (2 * (well_width_m / 2)**2))
        V_total += well

    # Ajouter les barrières personnalisées
    barrier_heights = [barrier_12, barrier_23, barrier_34]
    barrier_widths_nm = [width_12, width_23, width_34]
    barrier_widths_m = [w * 1e-9 for w in barrier_widths_nm]

    for i in range(3):
        center = (dot_positions[i] + dot_positions[i+1]) / 2
        sigma = (barrier_widths_m[i] / 2)**2
        barrier = barrier_heights[i] * np.exp(-((x_vals - center)**2) / (2 * sigma))
        V_total += barrier

    return V_total

def generate_dot_positions_from_barrier_widths(
    width_12, width_23, width_34,
    well_spacing_nm=10, center_nm=0
):
    """
    Génère automatiquement les positions des 4 QDots en fonction des largeurs de barrières entre eux.
    """
    # Convertir en mètres
    widths_nm = [width_12, width_23, width_34]
    widths_m = [w * 1e-9 for w in widths_nm]
    well_spacing_m = well_spacing_nm * 1e-9
    
    # Construction des positions (dot1 à dot4)
    dot_positions = []
    current_pos = center_nm * 1e-9 - sum(widths_m)/2 - 1.5 * well_spacing_m  # point de départ à gauche

    dot_positions.append(current_pos)
    for w in widths_m:
        current_pos += w + well_spacing_m  # largeur de barrière + espacement fixe
        dot_positions.append(current_pos)

    return dot_positions


# Parameters:
# time
T_final = 1e-12
nbr_pts = 1000
time_array = np.linspace(0, T_final, nbr_pts)
# meshgrid
x_vals = np.linspace(-100e-9, 100e-9, 2000)
x_nm = x_vals * 1e9  # for a display in nm
# barrier width
width_12 = 5
width_23 = 10
width_34 = 5


# Function calls:
# Generate the 4 QDots positions automaticaly
dot_positions = generate_dot_positions_from_barrier_widths(
    width_12=width_12,
    width_23=width_23,
    width_34=width_34,
    well_spacing_nm=5  # optional space between barrier and dot
)

# Generate a potential pulse at a given time
U_imp = pulse_U(time_array, t_start=2e-13, delta_t=1e-13, delta_U=65)
pot = potential_over_time(U_imp, x_vals,
    dot_positions,
    well_depth=30,
    well_width=10,
    barrier_12=20, barrier_23=80, barrier_34=20,
    width_12=5, width_23=5, width_34=5
)


# Visualization:
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Initial graph
t0 = 0
line, = ax.plot(x_vals, pot[t0], lw=2)
ax.set_title(f"Potentiel à t = {time_array[t0]*1e15:.1f} fs")
ax.set_xlabel("x (m)")
ax.set_ylabel("Potentiel V(x)")

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
slider = Slider(
    ax=ax_slider,
    label='Temps (index)',
    valmin=0,
    valmax=len(time_array) - 1,
    valinit=t0,
    valstep=1,
)

# Callback for update
def update(val):
    idx = int(slider.val)
    line.set_ydata(pot[idx])
    ax.set_title(f"Potentiel à t = {time_array[idx]*1e15:.1f} fs")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()