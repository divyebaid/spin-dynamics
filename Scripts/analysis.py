# -*- coding: utf-8 -*-

"""
This module provides functions to analyze and visualize the different aspects of a system described by the Hubbard model.
It includes functions to compute meshgrids for U and t values, plot contour plots, analyze ratios of energies, and visualize potentials.
It also includes a function to compute the tunneling splitting in a double well potential.
"""

# Importing necessary libraries


import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

from tqdm.auto import tqdm
from utils import *
from scipy.special import erf
from scipy.signal import find_peaks
from scipy.linalg import eigh_tridiagonal


# Functions


def meshgrid_u_t(filename: str, u_min: float = -4, u_max: float = -3, t_min: float = -3.5, t_max: float = 1.0, u_points: int = 20, t_points: int = 45, fenetre = 3):

    """
    This function computes the meshgrid for U and t values, calculates the top Hubbard states, and saves the results in a .npz file.

    Parameters:
    - filename : str : name of the output file (without extension)
    - u_min : float : minimum value for U (in log scale, default: -4)
    - u_max : float : maximum value for U (in log scale, default: -3)
    - t_min : float : minimum value for t (in log scale, default: -3.5)
    - t_max : float : maximum value for t (in log scale, default: 1.0)
    - u_points : int : number of points for U (default: 20)
    - t_points : int : number of points for t (default: 45)
    - fenetre : int : time window for the top Hubbard states calculation (default: 3)

    Returns:
    - None : saves the results in a .npz file with the specified filename
    """

    # Grid
    u_vals = np.logspace(u_min, u_max, u_points)
    t_vals = np.logspace(t_min, t_max, t_points)
    uu, tt = np.meshgrid(u_vals, t_vals)

    # Initialisation
    P1     = np.zeros_like(uu)
    T1     = np.zeros_like(uu)
    Max_T  = np.zeros_like(uu)
    Max_P  = np.zeros_like(uu)
    t_matrix_base = get_hopping_simple_matrix(4, 1)

    # Calculations
    for i in tqdm(range(t_points)):
        for j in range(u_points):
            U_ij     = uu[i, j]
            t_mat_ij = tt[i, j] * t_matrix_base
            temps = sc.hbar * 15*fenetre / (tt[i, j] * sc.e)  
            T, Tmp, _ = top_hubbard_states(
                T=temps,
                U=U_ij,
                t_matrix=t_mat_ij,
                display=False,
            )
            # Peak detection
            peaks, _ = find_peaks(Tmp[1])
            if peaks.size:
                # 1st peak
                idx1      = peaks[0]
                T1[i, j]  = T[idx1]
                P1[i, j]  = Tmp[1][idx1]
                # Max peak
                local_idx = np.argmax(Tmp[1][peaks])
                Max_T[i, j] = T[peaks[local_idx]]
                Max_P[i, j] = Tmp[1][peaks[local_idx]]
            else:
                T1[i, j] = np.nan
                P1[i, j] = np.nan
                Max_T[i, j] = np.nan
                Max_P[i, j] = np.nan

    # Save results
    np.savez(
        "data/"+filename+".npz",
        u_vals=u_vals,
        t_vals=t_vals,
        T1=T1,
        P1=P1,
        Max_T=Max_T,
        Max_P=Max_P,
    )
    print(f"Data saved in « {filename}.npz »")


def plot_meshgrid_from_file(filename: str, key: str, logscale: bool = False):

    """
    This function loads a .npz file and plots the specified key as a contour plot.

    Parameters:
    - filename : str : path to the .npz file containing the data
    - key : str : the key to plot (e.g., 'P1', 'T1', 'Max_P', 'Max_T', 'Max_P1')
    - logscale : bool : whether to use logarithmic scale for the colorbar (default: False)

    Returns:
    - None : displays the contour plot
    """

    #Load data
    data = np.load("data/"+filename+".npz")
    u_vals = data['u_vals']
    t_vals = data['t_vals']
    Z      = data[key]
    if logscale:
        Z = np.log10(Z)
        Z[np.isneginf(Z)] = np.nan  # Remplace -inf by NaN to avoid plotting issues
    U_mesh, T_mesh = np.meshgrid(u_vals, t_vals)

    # Plotting 
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(
        U_mesh,
        T_mesh,
        Z,
        levels=100,
        cmap='viridis',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(cp, label=key)
    plt.xlabel("U (eV)")
    plt.ylabel("t (eV)")
    plt.title(f"Contour de {key}")
    plt.show()


def plot_with_one_u(filename: str, key: str, logscale: bool = False):

    """
    This function loads a .npz file and plots the specified key against t/u_vals.

    Parameters:
    - filename : str : path to the .npz file containing the data
    - key : str : the key to plot (e.g., 'P1', 'T1', 'Max_P', 'Max_T', 'Max_P1')
    - logscale : bool : whether to use logarithmic scale for the y-axis (default: False)

    Returns:
    - None : displays the plot
    """
    data = np.load("data/"+filename+".npz")
    u_vals = data['u_vals']
    t_vals = data['t_vals']
    Z      = data[key]

    plt.figure(figsize=(8, 6))
    cp = plt.plot(t_vals/u_vals, Z, label=key)
    plt.xscale('log')
    if logscale:
        plt.yscale('log')
    plt.xlabel("t/U ")
    plt.ylabel(key)
    plt.title(f" {key}")
    plt.legend()
    plt.show()


def T_sur_T1(filename: str, tol: float = 1e-2, min_length: int = 5, logscale: bool = False):

    """
    This function analyzes the ratio Max_T / T1 from a .npz file.
    It identifies and plots the plateaus in the ratio, indicating regions of interest.

    Parameters:
    - filename : str : path to the .npz file containing the data
    - tol : float : tolerance for detecting plateaus (default: 1e-2)
    - min_length : int : minimum length of plateau segments to consider (default: 5)
    - logscale : bool : whether to use logarithmic scale for the y-axis (default: False)

    Returns:
    - None : displays the plot and prints the plateau values
    """

    # Load data
    data   = np.load(f"data/{filename}.npz")
    u0     = data['u_vals'][0]
    t_vals = data['t_vals']
    R      = data['Max_T'][:,0] / data['T1'][:,0]

    # Prepare and sort data
    x = t_vals / u0
    order = np.argsort(x)
    x = x[order]
    y = R[order]

    # Calculate differential of y with respect to log(x)
    ln_x    = np.log(x)
    dy_dlnx = np.diff(y) / np.diff(ln_x)
    mask = np.concatenate(([False], np.abs(dy_dlnx) < tol))

    # Identify segments of the mask where the condition is True (length >= min_length)
    segments = []
    i, N = 0, len(mask)
    while i < N:
        if mask[i]:
            start = i
            while i < N and mask[i]:
                i += 1
            end = i
            if end - start >= min_length:
                segments.append((start, end))
        else:
            i += 1

    if not segments:
        print("Aucun plateau détecté.")
        return

    # We keep only the last three segments
    derniers = segments[-3:]

    # Plotting
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(8,5))
    plt.plot(x, y, color=colors[0], label="Max_T / T1")

    for k, (s, e) in enumerate(derniers, start=1):
        mean_val = np.nanmean(y[s:e])
        print(f"P{k} (t/U ∈ [{x[s]:.2e}, {x[e-1]:.2e}]) ≃ {mean_val:.3f}")
        plt.hlines(mean_val, x[s], x[e-1],
                   colors=colors[k],
                   linestyles='dashed',
                   linewidth=2,
                   label=f"P{k} ≃ {mean_val:.2f}")

    plt.xscale('log')
    if logscale:
        plt.yscale('log')
    plt.xlabel('t / U')
    plt.ylabel('Max_T / T1')
    plt.title('Rapport et plateaux détectés')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def draw_potential(a=1.276, sigma = 10, b=4, x_vals = np.linspace(-100e-9, 100e-9, 10000), display=True):

    """
    This function computes and optionally displays the electrostatic potential defined by a quadratic term and an exponential term.

    Parameters:
    - a : coefficient for the quadratic term (in meV/nm^2)
    - sigma : standard deviation for the Gaussian term (in nm)
    - b : coefficient for the exponential term (in meV)
    - x_vals : array of positions (in m) where the potential is computed
    - display : boolean to control whether to display the potential plot
    Returns:

    - x_nm : positions in nanometers
    - V_expr_meV : computed potential in meV
    """

    x_nm = x_vals * 1e9
    
    V_expr = a*5e11*x_vals**2 + b*1e-3*np.exp(-((x_vals)**2)/(4*(sigma*1e-9)**2)) #(eV)
    V_expr_meV = V_expr * 1e3 # conversion in meV

    # Affichage
    if display:
        plt.figure(figsize=(8, 5))
        plt.plot(x_nm, V_expr_meV, label=r"$V(x)$ (nouvelle expression)", color='teal')
        plt.xlabel("Position x (nm)")
        plt.ylabel("Potentiel $V(x)$ (meV)")
        plt.title("Potentiel électrostatique défini par erreur et exponentielle")
        #plt.plot(x_vals * 1e9, (V_expr - V_expr[::-1]) / sc.e * 1e3, color='red', label=r"$V(x) - V(-x)$ (symétrie)", linestyle='--')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return x_nm, V_expr_meV


# Unfinished
def compute_hopping(a=1.276, sigma = 6 ,b=4, m_eff=0.067 * sc.m_e, N=2000, L=100e-9, plot=True):

    """
    This function computes the tunneling splitting in a double well potential defined by a quadratic term and an exponential term.

    Parameters:
    a : coefficient for the quadratic term (in meV/nm^2)
    d : distance between the two wells (in nm)
    m_eff : effective mass of the electron (in kg)
    N : number of points in the discretization of the potential
    L : half-width of the potential well (in nm)
    plot : boolean to control whether to display the potential and wavefunctions

    Returns:
    delta_E : tunneling splitting energy (in meV)
    T_tunnel : tunneling time (in ps)
    """

    x_vals = np.linspace(-L, L, N)
    dx = x_vals[1] - x_vals[0]

    V_x = draw_potential(a, sigma,b, x_vals, display=False)[1]*1e-3 * sc.e # En Joules

    # Kinetic hamiltonian
    kin_diag = np.full(N, sc.hbar**2 / (m_eff * dx**2))
    off_diag = np.full(N - 1, -sc.hbar**2 / (2 * m_eff * dx**2))

    # Total Hamiltonian
    H_diag = kin_diag + V_x
    e_vals, e_vecs = eigh_tridiagonal(H_diag, off_diag)

    # meV energies
    e0, e1 = e_vals[0:2]
    delta_E = (e1 - e0) / sc.e * 1e3  # en meV

    # Temps tunnel en ps
    T_tunnel = sc.h / (e1 - e0) * 1e12  # en ps

    print(f"Ecart centre des qdots: {2*x_vals[np.argmin(V_x)]*1e9:.2f} nm")
    print(f"t: {np.sqrt((get_U(40e-9)*delta_E)/4)} meV")

    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(x_vals * 1e9, V_x /sc.e * 1e3, label="V(x)", color='black', lw=1)
        plt.plot(x_vals * 1e9, np.full(N,e0)*1e3/sc.e, label=r"$E_0$", alpha=0.6)
        plt.plot(x_vals * 1e9, np.full(N,e1)*1e3/sc.e, label=r"$E_1$", alpha=0.6)
        plt.xlabel("x (nm)")
        plt.ylabel("Énergie [meV]")
        plt.title(f"Splitting tunnel : ΔE = {delta_E:.3f} meV, T = {T_tunnel:.1f} ps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return delta_E, T_tunnel


temps = sc.hbar * 15*102 / (0.0394e-3 * sc.e)  
T, Tmp, _ = top_hubbard_states(
    T=temps,
    U=2.45e-3,
    t_matrix=0.0394e-3*get_hopping_simple_matrix(4,1),
    display=True,
)


compute_hopping(a=1.276, sigma = 6,b=20, m_eff=0.067 * sc.m_e, N=2000, L=100e-9, plot=True)