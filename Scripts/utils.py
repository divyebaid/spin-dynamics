# -*- coding: utf-8 -*-

"""
This module contains functions to create and manipulate quantum states, including creation and annihilation operators, time evolution, and transition probabilities. It is designed for use in quantum mechanics simulations, particularly in the context of spin systems.
"""

#Importing necessary libraries

import numpy as np
import numpy as np
import itertools
import scipy.linalg

# Functions

def create_bit_strings(N):
    """
    create all N-bit binary strings

    N: int, number of site
    ret_array: array of int, list of all arrangements 
    """
    ret_arr = list(itertools.product([0,1], repeat=N))
    ret_arr = np.array([list(arr) for arr in ret_arr])

    return ret_arr


def creation(state, i, spin):
    """
    Creation operator acting on spin of site i (0-indexed) of state

    state: array, state of the system
    i: int, site wanted
    spin: char, "u" or "d"
    """
    idx = 2 * i if spin == 'u' else 2*i + 1
    S_i = np.abs(np.sum(state[0:idx]))
    sign_factor = (-1) ** S_i

    if not state.any():
        return np.zeros(len(state))
    elif np.abs(state[idx]) == 1:
        return np.zeros(len(state))
    else:
        ret_state = state.copy()
        ret_state[idx] = 1
        return sign_factor*ret_state
    

def annihilation(state, i, spin):
    """
    Annihilation operator acting on spin of site i (0-indexed) of state

    state: array, state of the system
    i: int, site wanted
    spin: char, "u" or "d"
    """

    idx = 2 * i if spin == 'u' else 2*i + 1
    S_i = np.abs(np.sum(state[0:idx]))
    sign_factor = (-1) ** S_i
    if not state.any():
        return np.zeros(len(state))
    elif state[idx] == 0:
        return np.zeros(len(state))
    else:
        ret_state = state.copy()
        ret_state[idx] = 0
        return sign_factor*ret_state
    

def number_operator(state, i, spin):
    """
    Number operator acting on spin of site i (0-indexed) of state. Returns 0 or 1 whether the site i have the spin wanted or not

    state: array, state of the system
    i: int, site wanted
    spin: char, "u" or "d"
    """

    idx = 2 * i if spin == 'u' else 2 * i + 1
    return int(state[idx])
    

def time_evol_state(H, T, u, hbar=1):
    """
    Returns an array of statevectors corresponding to the time evolution of u under H, according to |u(t)> = exp(iHt/h)|u(0)>.

    H: array, hamiltonien of the system
    T: array, times of the system, often a np.linspace
    u: array, state considered
    hbar: float, reduced Planck constant (default = 1)
    returns: array, array of statevectors over time
    """

    return np.array([(time_evol_operator(H, t, hbar) @ u) for t in T])


def transition_probability_over_time(left_state, right_states):
    """
    This function returns an array corresponding to |<left_state|right_state>|^2 over time. It assumes right_states is an array of the T statevectors over time

    left_state: array, reference state of the system
    right_states: array, array of statevectors over time

    returns: array, array of probabilities of the left_state over time
    """
    
    
    ret_component = np.array([(np.vdot(left_state, right_state)) for right_state in right_states])
    ret_component = np.square(np.absolute(ret_component))

    return ret_component


def generate_base_states(N):
    """
    Generates the simple basis states of length N (e.g. [1,0], [0,1]) for a system of N sites

    N: int, number of sites
    Returns: array, array of the simple basis states
    """

    return np.eye(N)


def time_evol_operator(H, t, hbar=1):
    """
    This function returns the time evolution operator U for a given Hamiltonian H and time t. U = exp(-iHt/hbar)

    H: array, hamiltonian of the system
    t: float, time of the system
    returns: array, time evolution operator
    """

    return scipy.linalg.expm(-1j * H * t / hbar)


def prob_over_time(H,T,u,v,transpose = True):
    """
    Returns the transition probability over time T for a given Hamiltonian H, initial state u, and observed state v.
    The function computes the time evolution of the state u under the Hamiltonian H and then calculates the transition probability to the state v.
    H: array, Hamiltonian of the system
    T: array, time points at which to evaluate the transition probability
    u: array, initial state of the system
    v: array, observed state of the system
    returns: array, transition probability over time
    """

    if transpose: 
        u = np.atleast_2d(u).T
    # Ensure u is a column vector

    U = time_evol_state(H,T,u)
    ret_component = transition_probability_over_time(v,U)
    return ret_component


def hopping_term_sign_factor(state, i, k, spin):
    """
    This function returns the sign factor for the hopping term in the Hamiltonian.

    state: array, state of the system
    i: int, site index of the initial state
    k: int, site index of the final state
    spin: char, "u" or "d"

    returns: int, sign factor for the hopping term
    """

    # Hopping is equivalent to annihilation at i and creation at k
    # The sign factor is (-1)^(S_i + S_k), where S_i and S_k are the number of spins at sites i and k respectively

    idx_i = 2 * i if spin == 'u' else 2*i + 1
    idx_k = 2 * k if spin == 'u' else 2*k + 1

    S_i = np.abs(np.sum(annihilation(state, k, spin)[0:idx_i]))
    S_k = np.abs(np.sum(state[0:idx_k]))

    return (-1) ** (S_i + S_k)


def get_hubbard_states(N):
    """
    Generates all possible Hubbard states for a system of N sites.
    Each state is represented as a binary array of length 2N, where the first N bits represent spin-up electrons and the last N bits represent spin-down electrons.

    N: int, number of sites
    returns: array, array of all possible Hubbard states
    """
    dim = 2 * N  # 2 états (↑ et ↓) par site
    all_states = []

    # On choisit N positions parmi 2N pour y mettre les électrons (1s)
    for occ_indices in itertools.combinations(range(dim), N):
        state = np.zeros(dim, dtype=int)
        state[list(occ_indices)] = 1
        all_states.append(state)

    return np.array(all_states)


def hubbard_hamiltonian_matrix(N, t, U):
    """
    Returns the Hubbard Hamiltonian matrix for a system of N sites.
    
    N: int, number of sites
    t: array, hopping integral matrix (symmetric NxN matrix), t[i][j] represents the hopping amplitude between sites i and j
    U: float, on-site interaction strength
        
    returns: array, Hubbard Hamiltonian matrix in the basis of all possible states

    """
    states = get_hubbard_states(N)  # Get all possible Hubbard states
    dim = len(states)  # Dimension of the Hilbert space
    H = np.zeros((dim, dim))
    
    # Loop over all states (rows)
    for i in range(dim):
        state_i = states[i]
        
        # Loop over all states (columns)
        for j in range(dim):
            state_j = states[j]
            
            # Diagonal elements: Coulomb interaction term 
            if i == j:
                for site in range(N):
                    # Check if both up and down spins are present at the site
                    n_up = number_operator(state_i, site, 'u')
                    n_down = number_operator(state_i, site, 'd')
                    H[i, j] += U * n_up * n_down
            
            # Off-diagonal: Hopping terms
            else:
                # Determine if states i and j differ by a single hopping event
                for site1 in range(N):
                    for site2 in range(N):
                        if site1 != site2:
                            for spin in ['u', 'd']:
                                # Check if state_j can be obtained from state_i by hopping
                                # Apply annihilation followed by creation to attempt a hopping
                                temp_state = annihilation(state_i, site1, spin)
                                
                                # Only proceed if annihilation and creation are both successful
                                if np.any(temp_state):
                                    final_state = creation(temp_state, site2, spin)
                                    
                                    if np.any(final_state) and np.array_equal(np.abs(final_state), state_j):
                                        # Use the dedicated function to calculate the sign factor
                                        sign = hopping_term_sign_factor(state_i, site1, site2, spin)
                                        # Add the hopping term
                                        H[i, j] -= t[site1][site2] * sign
    
    return H


def hubbard_hamiltonian(state, t, U, prod_state):
    # retrns array [statevector of H_Hubbard * state, <prod_state| H_Hubbard |state>]
    # t is a symmetric matrix of overlap integrals (positive values)
    # U is a constant corresponding to intra-site Coulomb interaction

    N = int(len(state) / 2)
    coulomb_term = np.zeros(len(state))
    hopping_term = np.zeros(len(state))
    inner_product = 0

    keep_term = False
    if (len(state) == 4 and len(prod_state) == 4):
        if (state.tolist() in two_hubbard_states.tolist() and prod_state.tolist() in two_hubbard_states.tolist()):
            keep_term = True
    elif (len(state) == 8 and len(prod_state) == 8):
        if (state.tolist() in four_hubbard_states.tolist() and prod_state.tolist() in four_hubbard_states.tolist()):
            keep_term = True

    if keep_term:
        for i in range(N):
            add_coulomb = U * number_operator(state, i, 'u') * number_operator(state, i, 'd') * state
            inner_product = inner_product + state_inner_prod(prod_state, add_coulomb)
            coulomb_term = coulomb_term + add_coulomb
            for k in range(N):
                if i != k:
                    for spin in spins:
                        sign_a = hopping_term_sign_factor(state, i, k, spin)
                        sign_b = hopping_term_sign_factor(state, k, i, spin)
                        add_hopping = (-1) * t[i][k] * (sign_a * np.abs(creation(np.abs(annihilation(state, k, spin)), i, spin)) 
                                                        + sign_b * np.abs(creation(np.abs(annihilation(state, i, spin)), k, spin)))
                        inner_product = inner_product + state_inner_prod(prod_state, add_hopping)
                        hopping_term = hopping_term + add_hopping
        
        res_state = coulomb_term + hopping_term
        
        return [res_state, inner_product]
    else:
        return [state, 0]

def hubbard_hamiltonian_bis(state, t, U, prod_state):
    # retrns array [statevector of H_Hubbard * state, <prod_state| H_Hubbard |state>]
    # t is a symmetric matrix of overlap integrals (positive values)
    # U is a constant corresponding to intra-site Coulomb interaction

    N = int(len(state) / 2)
    coulomb_term = np.zeros(len(state))
    hopping_term = np.zeros(len(state))
    inner_product = 0

    keep_term = True

    if keep_term:
        for i in range(N):
            add_coulomb = U * number_operator(state, i, 'u') * number_operator(state, i, 'd') * state
            inner_product = inner_product + state_inner_prod(prod_state, add_coulomb)
            coulomb_term = coulomb_term + add_coulomb
            for k in range(N):
                if i != k:
                    for spin in spins:
                        sign_a = hopping_term_sign_factor(state, i, k, spin)
                        sign_b = hopping_term_sign_factor(state, k, i, spin)
                        add_hopping = (-1) * t[i][k] * (sign_a * creation(annihilation(state, k, spin), i, spin) 
                                                        + sign_b * creation(annihilation(state, i, spin), k, spin))
                        inner_product = inner_product + state_inner_prod(prod_state, add_hopping)
                        hopping_term = hopping_term + add_hopping
        
        res_state = coulomb_term + hopping_term
        
        return [res_state, inner_product]
    else:
        return [state, 0]

def state_inner_prod(state_one, state_two):
    # inner product operation on states (assuming orthonormal states)
    # this function is used only for <state_one| operator |state_two> thus returning (state_two / state_one) 

    with np.errstate(divide='ignore', invalid='ignore'):
        quotient = np.true_divide(state_two, state_one)
        quotient[~np.isfinite(quotient)] = np.nan

    if state_one.any() and state_two.any():
        first_multiple = quotient[np.isfinite(quotient)][0]
        if np.all(np.isnan(quotient) | (quotient == first_multiple)):
            ret_val = first_multiple
        else:
            ret_val = 0
    else:
        ret_val = 0

    return ret_val