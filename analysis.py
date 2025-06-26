# %%
# -*- coding: utf-8 -*-

"""
This module provides functions to analyze and visualize the different aspects of a system described by the Hubbard model.
It includes functions to compute meshgrids for U and t values, plot contour plots, analyze ratios of energies, and visualize potentials.
It also includes a function to compute the tunneling splitting in a double well potential.
"""

# Importing necessary libraries


from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as sc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

from tqdm.auto import tqdm
from utils import *
from scipy.special import erf
from scipy.signal import find_peaks
from scipy.linalg import eigh_tridiagonal
from utils_c import top_hubbard_states_c

from new_utils import *


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


def plot_Ut_from_file(filename: str, key: str, logscale: bool = False):

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
    
    V_expr = a*5e11*x_vals**2 + b*1e-3*np.exp(-((x_vals)**2)/(2*(sigma*1e-9)**2)) #(eV)
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


def compute_hopping(a=15, sigma = 10 ,b=10, m_eff=0.19 * sc.m_e, N=200, L=70e-9, plot=True):

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
    e0, e1,e2 = e_vals[0:3]
    delta_E = (e1 - e0) / sc.e * 1e3  # en meV
    U_bis = (e2 - e0) / sc.e *1e3 # en ps
    t_hopping = np.sqrt((U_bis*delta_E)/4)
    dist = np.abs(2*x_vals[np.argmin(V_x)]*1e9)

    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(x_vals * 1e9, V_x /sc.e * 1e3, label="V(x)", color='black', lw=1)
        plt.plot(x_vals * 1e9, np.full(N,e0)*1e3/sc.e, label=r"$E_0$", alpha=0.6)
        plt.plot(x_vals * 1e9, np.full(N,e1)*1e3/sc.e, label=r"$E_1$", alpha=0.6)
        plt.plot(x_vals * 1e9, np.full(N,e_vals[2])*1e3/sc.e, label=r"$E_2$", alpha=0.6)
        plt.xlabel("x (nm)")
        plt.ylabel("Énergie [meV]")
        plt.title(f"Splitting tunnel : ΔE = {delta_E:.3f} meV, t = {np.sqrt((U_bis*delta_E)/4)} meV)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return U_bis*1e-3, t_hopping*1e-3, dist, delta_E


def meshgrid_b_sigma_ut(filename: str, b_min: float = 0, b_max: float = 20, sigma_min: float = 1, sigma_max: float = 20, b_points: int = 4, sigma_points: int = 4, fenetre = 10):

    """
    This function computes the meshgrid for b and sigma values, calculates the top Hubbard states, and saves the results in a .npz file.

    Parameters:
    - filename : str : name of the output file (without extension)
    - b_min : float : minimum value for b (in log scale, default: -4)
    - b_max : float : maximum value for b (in log scale, default: -3)
    - sigma_min : float : minimum value for sigma (in log scale, default: -3.5)
    - sigma_max : float : maximum value for sigma (in log scale, default: 1.0)
    - b_points : int : number of points for b (default: 20)
    - sigma_points : int : number of points for sigma (default: 45)
    - fenetre : int : time window for the top Hubbard states calculation (default: 3)

    Returns:
    - None : saves the results in a .npz file with the specified filename
    """

    # Grid
    b_vals = np.linspace(b_min, b_max, b_points)
    sigma_vals = np.linspace(sigma_min, sigma_max, sigma_points)
    bb, ss = np.meshgrid(b_vals, sigma_vals)

    # Initialisation
    U     = np.zeros_like(bb)
    t     = np.zeros_like(bb)
    dist   = np.zeros_like(bb)
    t_matrix_base = get_hopping_simple_matrix(4, 1)

    # Calculations
    for i in tqdm(range(b_points)):
        for j in range(sigma_points):
            U[i,j], t[i,j], dist[i,j] = compute_hopping(b=b_vals[i], sigma=sigma_vals[j], a=15, m_eff=0.19 * sc.m_e, plot=False)[0:3]
    # Save results
    np.savez(
        "data/"+filename+".npz",
        b_vals=b_vals,
        sigma_vals=sigma_vals,
        U=U,
        t=t,
        dist=dist
    )
    print(f"Data saved in « {filename}.npz »")


def plot_bsigma_from_file(filename: str, key: str, logscale: bool = False):

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
    b_vals = data['b_vals']
    sigma_vals = data['sigma_vals']
    Z      = data[key]
    if logscale:
        Z = np.log10(Z)
        Z[np.isneginf(Z)] = np.nan  # Remplace -inf by NaN to avoid plotting issues
    b_mesh, sigma_mesh = np.meshgrid(b_vals, sigma_vals)

    # Plotting 
    plt.figure(figsize=(8, 6))
    if key in ['U', 't']:
        cp = plt.contourf(
        b_mesh,
        sigma_mesh,
        Z*1e3,
        levels=100,
        cmap='viridis',
    )
    else:
        cp = plt.contourf(
            b_mesh,
            sigma_mesh,
            Z,
            levels=100,
            cmap='viridis',
        )
    plt.colorbar(cp, label=key)
    plt.xlabel("b (meV)")
    plt.ylabel("sigma (nm)")
    plt.title(f"Contour de {key}")
    plt.show()


def plot_bsigma_U_t2(filename: str):

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
    b_vals = data['b_vals']
    sigma_vals = data['sigma_vals']
    U      = data['U']
    t      = data['t']
    b_mesh, sigma_mesh = np.meshgrid(b_vals, sigma_vals)

    # Plotting 
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(
        b_mesh,
        sigma_mesh,
        np.log10(t**2/U),
        levels=100,
        cmap='viridis',
    )
    plt.colorbar(cp)
    plt.xlabel("b (meV)")
    plt.ylabel("sigma (nm)")
    plt.title(r"$t^2/U$ (logscale)")
    plt.show()


#matplotlib.use("Qt5Agg")  # Affichage interactif dans une fenêtre externe



def show_slider_peaks(filename: str, key: str, max_peaks: int = 5, logscale: bool = False):
    """
    Displays a slider to explore up to max_peaks peaks of raw data from .npz file,
    with a colorbar fixed between 0 and 1 (or between min/max of your data if you préfèr).
    """

    # Load data
    data = np.load(f"data/{filename}.npz", allow_pickle=True)
    raw = data[key]
    b_vals = data.get("b_vals")
    sigma_vals = data.get("sigma_vals")

    nb_b, nb_sigma = raw.shape
    Z_peaks = np.full((nb_b, nb_sigma, max_peaks), np.nan)
    for i in range(nb_b):
        for j in range(nb_sigma):
            arr = raw[i, j]
            if arr is not None and len(arr) > 0:
                n = min(len(arr), max_peaks)
                vals = np.asarray(arr)
                if logscale:
                    mask = vals > 0
                    vals = np.where(mask, np.log10(vals), np.nan)
                Z_peaks[i, j, :n] = vals[:n]

    B, Sigma = np.meshgrid(b_vals, sigma_vals, indexing='ij')

    # Colorbar fixée sur [0, 1]
    vmin = 0
    vmax = np.nanmax(Z_peaks)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, 'Peak #', 0, Z_peaks.shape[2]-1, valinit=0, valstep=1)

    # Premier affichage
    Z = Z_peaks[:, :, 0]
    cp = ax.contourf(Sigma, B, Z, levels=100, cmap=cmap, norm=norm)
    ax.set_xlabel('Sigma (nm)')
    ax.set_ylabel('B (meV)')
    maxZ = np.nanmax(Z)
    ax.set_title(f"{key}: Peak 0 (max={maxZ:.4g})")

    # Colorbar indépendante
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label(f"{key}")

    def draw(idx):
        Z = Z_peaks[:, :, idx]
        while ax.collections:
            ax.collections[0].remove()
        cp = ax.contourf(Sigma, B, Z, levels=100, cmap=cmap, norm=norm)
        maxZ = np.nanmax(Z)
        ax.set_title(f"{key}: Peak {idx} (max={maxZ:.4g})")
        fig.canvas.draw_idle()

    def on_slide(val):
        idx = int(val)
        draw(idx)

    slider.on_changed(on_slide)
    plt.show()


def meshgrid_b_sigma(filename: str, b_min: float = 0.5, b_max: float = 20, sigma_min: float =0.5, sigma_max: float = 20, b_points: int = 10, sigma_points: int = 10, fenetre = 1):

    """
    This function computes the meshgrid for b and sigma values, calculates the top Hubbard states, and saves the results in a .npz file.

    Parameters:
    - filename : str : name of the output file (without extension)
    - b_min : float : minimum value for b (in log scale, default: -4)
    - b_max : float : maximum value for b (in log scale, default: -3)
    - sigma_min : float : minimum value for sigma (in log scale, default: -3.5)
    - sigma_max : float : maximum value for sigma (in log scale, default: 1.0)
    - b_points : int : number of points for b (default: 20)
    - sigma_points : int : number of points for sigma (default: 45)
    - fenetre : int : time window for the top Hubbard states calculation (default: 3)

    Returns:
    - None : saves the results in a .npz file with the specified filename
    """

    # Grid
    b_vals = np.linspace(b_min, b_max, b_points)
    sigma_vals = np.linspace(sigma_min, sigma_max, sigma_points)
    bb, ss = np.meshgrid(b_vals, sigma_vals)

    # Initialisation
    P1     = np.zeros_like(bb)
    T1     = np.zeros_like(bb)
    Max_T  = np.zeros_like(bb)
    Max_P  = np.zeros_like(bb)
    dist   = np.zeros_like(bb)
    P_peaks  = np.zeros_like(bb, dtype=object)  # To store peak indices
    T_peaks  = np.zeros_like(bb, dtype=object)  # To store peak indices
    t_matrix_base = get_hopping_simple_matrix(4, 1)

    # Calculations
    for i in tqdm(range(b_points)):
        for j in range(sigma_points):
            U_ij, t_mat_ij, dist = compute_hopping(b=b_vals[i], sigma=sigma_vals[j], a=15, m_eff=0.19 * sc.m_e, plot=False)[0:3]
            temps = sc.hbar * 15*fenetre / (t_mat_ij * sc.e)  

            #t_mat_ij = t_mat_ij * t_matrix_base

            T, Tmp, _ = top_hubbard_states_c(
                T_final=temps,
                U=U_ij,
                t_hopping=t_mat_ij,
            )
            # Peak detection
            peak, _ = find_peaks(Tmp[1])
            P_peaks[i, j] = Tmp[1][peak]  # Store peak proba for later use
            T_peaks[i, j] = T[peak]  # Store peak indices for later use
            if peak.size:
                # 1st peak
                idx1      = peak[0]
                T1[i, j]  = T[idx1]
                P1[i, j]  = Tmp[1][idx1]
                # Max peak
                local_idx = np.argmax(Tmp[1][peak])
                Max_T[i, j] = T[peak[local_idx]]
                Max_P[i, j] = Tmp[1][peak[local_idx]]
            else:
                T1[i, j] = np.nan
                P1[i, j] = np.nan
                Max_T[i, j] = np.nan
                Max_P[i, j] = np.nan

    # Save results
    np.savez(
        "data/"+filename+".npz",
        b_vals=b_vals,
        sigma_vals=sigma_vals,
        T1=T1,
        P1=P1,
        Max_T=Max_T,
        Max_P=Max_P,
        T_peaks=T_peaks,
        P_peaks=P_peaks,
        dist=dist
    )
    print(f"Data saved in « {filename}.npz »")



def meshgrid_b_sigma_save(filename: str, b_min: float = 0.5, b_max: float = 20, sigma_min: float = 0.5, sigma_max: float = 20, 
                         b_points: int = 10, sigma_points: int = 10, fenetre = 1, temp_dir="tmp_meshgrid_b_sigma"):

    # Create temp directory for intermediate saves
    os.makedirs(temp_dir, exist_ok=True)

    b_vals = np.linspace(b_min, b_max, b_points)
    sigma_vals = np.linspace(sigma_min, sigma_max, sigma_points)

    for i, b in enumerate(tqdm(b_vals)):
        # Variables pour cette ligne
        T1 = np.zeros(sigma_points)
        P1 = np.zeros(sigma_points)
        Max_T = np.zeros(sigma_points)
        Max_P = np.zeros(sigma_points)
        dist = np.zeros(sigma_points)
        P_peaks = np.empty(sigma_points, dtype=object)
        T_peaks = np.empty(sigma_points, dtype=object)

        for j, sigma in enumerate(sigma_vals):
            U_ij, t_mat_ij, dist_j = compute_hopping(b=b, sigma=sigma, a=15, m_eff=0.19 * sc.m_e, plot=False)[0:3]
            temps = sc.hbar * 15*fenetre / (t_mat_ij * sc.e)
            T, Tmp, _ = top_hubbard_states_c(
                T_final=temps,
                U=U_ij,
                t_hopping=t_mat_ij,
            )
            # Peaks
            peak, _ = find_peaks(Tmp[1])
            P_peaks[j] = Tmp[1][peak]
            T_peaks[j] = T[peak]
            if peak.size:
                idx1 = peak[0]
                T1[j] = T[idx1]
                P1[j] = Tmp[1][idx1]
                local_idx = np.argmax(Tmp[1][peak])
                Max_T[j] = T[peak[local_idx]]
                Max_P[j] = Tmp[1][peak[local_idx]]
            else:
                T1[j] = np.nan
                P1[j] = np.nan
                Max_T[j] = np.nan
                Max_P[j] = np.nan
            dist[j] = dist_j

        # Sauvegarder cette ligne dans un fichier temporaire
        np.savez(f"{temp_dir}/row_{i}.npz",
                 T1=T1, P1=P1, Max_T=Max_T, Max_P=Max_P,
                 dist=dist, P_peaks=P_peaks, T_peaks=T_peaks)

    # Assemblage final après toutes les lignes
    T1     = np.zeros((b_points, sigma_points))
    P1     = np.zeros((b_points, sigma_points))
    Max_T  = np.zeros((b_points, sigma_points))
    Max_P  = np.zeros((b_points, sigma_points))
    dist   = np.zeros((b_points, sigma_points))
    P_peaks = np.empty((b_points, sigma_points), dtype=object)
    T_peaks = np.empty((b_points, sigma_points), dtype=object)

    for i in range(b_points):
        f = np.load(f"{temp_dir}/row_{i}.npz", allow_pickle=True)
        T1[i, :]     = f["T1"]
        P1[i, :]     = f["P1"]
        Max_T[i, :]  = f["Max_T"]
        Max_P[i, :]  = f["Max_P"]
        dist[i, :]   = f["dist"]
        P_peaks[i, :] = f["P_peaks"]
        T_peaks[i, :] = f["T_peaks"]

    np.savez(
        "data/"+filename+".npz",
        b_vals=b_vals,
        sigma_vals=sigma_vals,
        T1=T1,
        P1=P1,
        Max_T=Max_T,
        Max_P=Max_P,
        T_peaks=T_peaks,
        P_peaks=P_peaks,
        dist=dist
    )
    print(f"Data saved in « {filename}.npz »")

    # (Optionnel : cleanup)
    # import shutil
    # shutil.rmtree(temp_dir)


def show_slider_peaks_1d(filename: str, key: str, max_peaks: int = 5):
    """
    Affiche un slider pour explorer les différents pics à sigma fixé à partir d'un fichier .npz.
    - filename : chemin du fichier .npz (sans .npz si dans "data/")
    - key : nom de la clé du tableau (ex : 'P_peaks')
    - max_peaks : nombre max de pics à slider
    """

    # Chargement des données
    data = np.load(f"data/{filename}.npz", allow_pickle=True)
    b_vals = data['b_vals']
    # On suppose que P_peaks est shape (N_b, N_peaks) ou (N_b, ?) si une seule valeur de sigma
    Z_peaks = data[key]
    # Si c'est un tableau d'objets (par exemple si créé avec dtype=object), on le convertit en float
    if isinstance(Z_peaks.flat[0], (np.ndarray, list)):
        # On suppose max_peaks correct
        N_b = len(b_vals)
        Y = np.full((N_b, max_peaks), np.nan)
        for i in range(N_b):
            arr = Z_peaks[i]
            arr = np.asarray(arr).flatten()
            n = min(len(arr), max_peaks)
            Y[i, :n] = arr[:n]
        Z_peaks = Y
    else:
        # Z_peaks déjà au bon format (N_b, N_peaks)
        pass

    vmin = 0
    vmax = np.nanmax(Z_peaks)

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.25)
    slider_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(slider_ax, 'Peak #', 0, max_peaks - 1, valinit=0, valstep=1)

    # Premier plot
    y = Z_peaks[:, 0]
    line, = ax.plot(b_vals, y, label=f'{key} [Peak 0]')
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel('b (meV)')
    ax.set_ylabel('Probabilité pic')
    ax.set_title(f"{key} – Peak 0 (max={np.nanmax(y):.4g})")
    ax.grid(True)
    ax.legend()

    def draw(idx):
        y = Z_peaks[:, idx]
        line.set_ydata(y)
        ax.set_title(f"{key} – Peak {idx} (max={np.nanmax(y):.4g})")
        fig.canvas.draw_idle()

    def on_slide(val):
        idx = int(val)
        draw(idx)

    slider.on_changed(on_slide)
    plt.show()



#meshgrid_b_sigma_save("sigma_fixe_bis",sigma_min = 6, sigma_max = 6, sigma_points = 1,b_points = 1000, fenetre = 40, temp_dir="tempo")
#draw_potential(a=1.276, sigma = 10, b=4, x_vals = np.linspace(-100e-9, 100e-9, 10000), display=True)
#draw_double_well_potential_corrected(a=1.276, sigma = 10, b=4, d=25, x_vals = np.linspace(-100e-9, 100e-9, 10000), display=True)