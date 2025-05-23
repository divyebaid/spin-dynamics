import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg

# Constants
# hbar = 1.05457E-34
hbar = 1
J = 1  # Simulated coupling constant (in Hz)
s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
s_z = np.array([[1, 0], [0, -1]])
s_all = [s_x, s_y, s_z]              
I = np.identity(2)


def generate_base_states(N):
    # returns the simple basis statevectors of length N (e.g. [1,0], [0,1]) 

    ret_states = np.empty(N, dtype=object)
    for n in range(N):
        add_state = np.zeros(N)
        add_state[n] = 1
        ret_states[n] = add_state

    return ret_states


def time_evol(H, t):
    # returns time evolution operator = exp(-i * H * t / hbar)
    hbar = 1
    ret_H = scipy.linalg.expm(-1j * H * t / hbar)

    return ret_H


def prob_over_time(H, T, u, v, transpose = True):
    # returns probability over time T, Hamiltonian H, initial state u, observed state v i.e. <v| exp(-iHt/hbar) |u>
    # set transpose = True if giving u as a row vector

    if transpose: 
        u = np.atleast_2d(u).T

    ret_component = np.array([(v @ time_evol(H, t) @ u)[0] for t in T])
    ret_component = np.square(np.absolute(ret_component))

    return ret_component


def time_evol_state(H, T, u):
    # returns an array of statevectors corresponding to the time evolution of u under H

    ret_component = np.array([(time_evol(H, t) @ u) for t in T])

    return ret_component


def prob_of_state(left_state, right_states):
    # returns an array corresponding to |<left_state|right_state>|^2 over time

    ret_component = np.array([(np.dot(left_state, right_state)) for right_state in right_states])
    ret_component = np.square(np.absolute(ret_component))

    return ret_component


def get_spin_operators_mat(N):
    # generates the set of spin operators for the Heisenberg spin chain model
    # returns a list l where l[i][a] = S_(i, a) for the a-th Pauli matrix for the i-th spin site

    identity_matrices = [np.identity(2**n) for n in range(N)]

    spin_operators_mat = []
    for n in range(N):
        spin_n = np.empty(3, dtype=object)
        left_identity = identity_matrices[n]
        right_identity = identity_matrices[N - n - 1]
        for m in range(3):
            s = s_all[m]
            if n == 0:
                s_n = np.kron(s, right_identity)
            elif n == N - 1:
                s_n = np.kron(left_identity, s)
            else:
                s_n = np.kron(np.kron(left_identity, s), right_identity)
            spin_n[m] = s_n
        spin_operators_mat.append(0.5 * spin_n)

    return spin_operators_mat


def simple_heisenberg_hamiltonian(N, J):
    # returns Heisenberg spin chain model Hamiltonian for N spins (single spin in single site basis)
    # assuming no external B field, assuming constant J

    dim = 2**N
    spin_operators_mat = get_spin_operators_mat(N)

    spin_dot_products = np.zeros((dim, dim))
    for n in range(N - 1):
        for m in range(3):
            spin_dot_products = spin_dot_products + np.matmul(spin_operators_mat[n][m], spin_operators_mat[n + 1][m])
        
    ret_Hamiltonian = (-1) * J * spin_dot_products

    return ret_Hamiltonian


def variable_J_heisenberg_hamiltonian(N, J):
    # returns Heisenberg spin chain model Hamiltonian for N spins (single spin in single site basis)
    # assuming no external B field but varying J (assuming spatial isotropy) and non-nearest neighbour interactions
    # J is strictly an upper triangular matrix of coupling constants

    dim = 2**N
    spin_operators_mat = get_spin_operators_mat(N)

    spin_dot_products = np.zeros((dim, dim))
    for i in range(N):
        for j in range(i + 1, N):
            coupling = J[i][j]
            for m in range(3):
                spin_dot_products = spin_dot_products + coupling * np.matmul(spin_operators_mat[i][m], spin_operators_mat[j][m])

    ret_Hamiltonian = (-1) * spin_dot_products
    
    return ret_Hamiltonian


def general_heisenberg_hamiltonian(N, J, B):
    # returns Heisenberg XXX spin chain model Hamiltonian for N spins (single spin in single site basis)
    # accounting for external B field, varying J and non-nearest neighbour interactions
    # J is strictly an upper triangular matrix of coupling constants
    # B is strictly a 3-vector of magnetic H-field (?)

    dim = 2**N
    g = 2.00  # g-factor for spin (to an approximation)
    mu_B = 1  # Bohr magneton
    spin_operators_mat = get_spin_operators_mat(N)
    H_spin = variable_J_heisenberg_hamiltonian(N, J)


    H_B = np.zeros((dim, dim))
    for n in range(N):
        for m in range(3):
            add_H = (-1) * g * mu_B * B[m] * spin_operators_mat[n][m]
            H_B = H_B + add_H

    ret_Hamiltonian = H_spin + H_B

    return ret_Hamiltonian


if __name__ == '__main__':
    # Plotting spin chain probabilities (basic model)
    hbar = 1 
    J = np.array([
        [0.00, 1.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
        [0.00, 0.00, 0.00, 0.00]
    ])
    B = np.array([0, 0, 0])
    num_e = 4  # number of spin sites being simulated
    T = np.linspace(0, 1000, 10000)
    H = general_heisenberg_hamiltonian(num_e, J, B)
    u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    plt.figure(figsize=(120, 6))
    time_state = time_evol_state(H, T, u)
    for n in range(2 ** num_e):
        v = generate_base_states(2 ** num_e)[n]
        res = prob_of_state(v, time_state)
        if res.any():
            bin_rep = format(n, f'0{num_e}b')
            spins_string = bin_rep.replace('0', '↑').replace('1', '↓')
            spins_string = '|' + spins_string + '>'
            plt.plot(T, res, label=spins_string)
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Probability')