# -*- coding: utf-8 -*-

#Importing necessary libraries

import numpy as np
import numpy as np
import itertools

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
    
    
    #number_state = creation(annihilation(state, i, spin), i, spin)
    #if state.any():
    #    if (np.abs(number_state) == state).all():
    #        return 1
    #    elif not number_state.any():
    #        return 0
    #    else:
    #        return 'error in number operator'
    #else:
    #    return 'error: passing empty state to number operator'
    

    idx = 2 * i if spin == 'u' else 2 * i + 1
    return int(state[idx])
    

def time_evol_state(H, T, u):
    """
    Returns an array of statevectors corresponding to the time evolution of u under H, according to |u(t)> = exp(iHt/h)|u(0)>.

    H: array, hamiltonien of the system
    T: array, times of the system, often a np.linspace
    u: array, state considered

    """


    ret_component = np.array([(time_evol(H, t) @ u) for t in T])

    return ret_component


def prob_of_state(left_state, right_states):
    """
    This function returns an array corresponding to |<left_state|right_state>|^2 over time. It assumes right_states is an array of the T statevectors over time

    left_state: array, reference state of the system
    right_states: array, array of statevectors over time

    returns: array, array of probabilities of the left_state over time
    """
    
    
    ret_component = np.array([(np.dot(left_state, right_state)) for right_state in right_states])
    ret_component = np.square(np.absolute(ret_component))

    return ret_component

