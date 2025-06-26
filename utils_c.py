import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as sc
from ctypes import c_int, c_longlong, c_double, POINTER, Structure, CDLL, byref, c_char_p, cast, pointer
from utils import get_hubbard_states, get_label, get_sampling_timestep, hubbard_hamiltonian_matrix, get_hopping_simple_matrix

# Constants
eV = 1.602176634e-19
HBAR = sc.hbar

class State(Structure):
    _fields_ = [
        ("size", c_int),
        ("occupancy", POINTER(c_int))
    ]

def load_c_library():
    """Load shared library containing top_hubbard_states_interface."""
    for name in ('C/libhubbard.so', 'C/libhubbard.dll'):
        try:
            lib = CDLL(name)
            if hasattr(lib, 'top_hubbard_states_interface'):
                lib.top_hubbard_states_interface.argtypes = [
                    c_int, c_double, c_double, c_int,
                    POINTER(State), c_char_p, c_double
                ]
                lib.top_hubbard_states_interface.restype = None
                return lib
        except OSError as e:
            continue
    raise FileNotFoundError("Library libhubbard not found. Please compile or install it.")
from ctypes import (
    Structure, POINTER, c_int, c_double, c_char_p,
    CDLL, byref, cast
)



def python_list_to_state(py_list):
    # 1) build a C array of ints
    ArrayType = c_int * len(py_list)
    c_arr = ArrayType(*py_list)

    # 2) wrap it in your State struct
    s = State()
    s.size = len(py_list)
    s.occupancy = cast(c_arr, POINTER(c_int))

    # 3) keep a reference alive so Python doesn't GC the array
    s._keep = c_arr

    # 4) DEBUG: verify the layout & contents

    return s


def top_hubbard_states_c(
    T_final=1e-11,
    U=1e-3,
    t_hopping=1e-1,
    nbr_pts=30000,
    top_n=4,
    init_state=[0, 1, 1, 0, 1, 0, 1, 0],
    csv_filename="top_hubbard_states.csv",
    npz_filename="top_hubbard_states.npz", 
    display = False
):
    """Run the pipeline: call C interface, convert CSV->NPZ, plot, and return results."""
    # Load C library
    lib = load_c_library()
    N = len(init_state)//2
    states = get_hubbard_states(N)
    U = U * eV
    t_hopping = t_hopping  * eV
    H = hubbard_hamiltonian_matrix(N, t_hopping*get_hopping_simple_matrix(N,1), U)
    states = get_hubbard_states(N)

    dt = get_sampling_timestep(H)
    nbr_pts   = int(T_final/dt) # nombre théorique de points
    nbr_pts   = min(1.2*nbr_pts, 500000)         
    if nbr_pts == 500000:       
        print(f"Trop long {nbr_pts}")
    # Prepare state and call the C function
    state = python_list_to_state(init_state)
    lib.top_hubbard_states_interface(
        N, U, T_final, int(nbr_pts),
        byref(state), csv_filename.encode('utf-8'), t_hopping
    )

    # Wait for CSV creation (max 5s)
    timeout = 5.0
    while timeout > 0 and not os.path.exists(csv_filename):
        time.sleep(0.1)
        timeout -= 0.1
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"CSV file not created: {csv_filename}")

    # Read CSV and remove it immediately
    df = pd.read_csv(csv_filename)
    os.remove(csv_filename)

    # Convert CSV to NPZ
    times = sorted(df['t_idx'].unique())
    data = {}
    for idx in df['idx'].unique():
        sub = df[df['idx'] == idx].sort_values('t_idx')
        proba = sub['proba'].astype(float).values
        arr = np.zeros(len(times), float)
        arr[:len(proba)] = proba
        data[f'state_{idx}'] = arr
    time_array = np.array(times, float)
    np.savez(npz_filename, time=time_array, **data)

    # Load NPZ and extract arrays
    npz = np.load(npz_filename)


    t = np.linspace(0, T_final, int(nbr_pts))


    loaded = {k: npz[k] for k in npz.files if k != 'time'}
    state_keys = sorted(loaded.keys(), key=lambda k: int(k.split('_')[1]))
    probs = np.array([loaded[k] for k in state_keys])  # (n_states, n_times)

    # Find the top_n states with max proba
    max_probs = probs.max(axis=1)
    top_idxs = np.argsort(max_probs)[::-1][:top_n]

    # Plot only the top states
    if display:
        plt.figure(figsize=(10, 6))
        for i in top_idxs:
            plt.plot(t, probs[i], label="|"+get_label(states[i])+">")
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.title(f'Top {top_n} Hubbard States')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Return (time_array, probabilities, indices, names)
    return t, probs[top_idxs], [states[i] for i in top_idxs]

