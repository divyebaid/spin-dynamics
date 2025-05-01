import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import scipy.linalg
import itertools
# import odeintw
import qutip as qt
import sympy as sp
import scipy.sparse
import scipy.sparse.linalg
from scipy.integrate import solve_ivp
from Heisenberg_model import time_evol, generate_base_states
from Hubbard_utils import *
from utils import *

# Utility
spins = ['u', 'd']  # spins up and down
two_hubbard_states = get_two_hubbard_states()
four_hubbard_states = get_four_hubbard_states()
four_hubbard_states_arrows = get_four_hubbard_states_arrows()
rearr_Sz_subspace = get_rearr_Sz_subspace()
four_T_S_states_arrows = get_four_T_S_states_arrows()


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
    

def hopping_term_sign_factor(state, i, k, spin):
    # returns the appropriate sign (+/- 1) to be used in the add_hopping term in the Hamiltonian
    # only for the operation creation(annihilation(state, k, spin), i, spin)

    idx_i = 2 * i if spin == 'u' else 2*i + 1
    idx_k = 2 * k if spin == 'u' else 2*k + 1
    S_i = np.abs(np.sum(annihilation(state, k, spin)[0:idx_i]))
    S_k = np.abs(np.sum(state[0:idx_k]))
    ret_factor = (-1) ** (S_i + S_k)

    return ret_factor
    

def hubbard_hamiltonian(state, t, U, prod_state):
    # returns array [statevector of H_Hubbard * state, <prod_state| H_Hubbard |state>]
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


def create_two_hubbard_hamiltonian(t, U):
    two_hubbard = np.zeros((16,16))
    dummy_states = create_bit_strings(4)
    for a in range(len(dummy_states)):
        for b in range(len(dummy_states)):
            two_hubbard[a][b] = hubbard_hamiltonian(dummy_states[b], t, U, dummy_states[a])[1]

    return two_hubbard


def create_four_hubbard_hamiltonian(t, U):
    four_hubbard = np.zeros((256,256))
    dummy_states = create_bit_strings(8)
    for a in range(len(dummy_states)):
        for b in range(len(dummy_states)):
            four_hubbard[a][b] = hubbard_hamiltonian(dummy_states[b], t, U, dummy_states[a])[1]

    return four_hubbard


def create_four_hubbard_hamiltonian_half_occ(t, U):
    four_hubbard = np.zeros((44, 44))
    dummy_states = four_hubbard_states
    for a in range(len(dummy_states)):
        for b in range(len(dummy_states)):
            four_hubbard[a][b] = hubbard_hamiltonian(dummy_states[b], t, U, dummy_states[a])[1]

    return four_hubbard


def computational_to_t_s_basis_change():
    # Returns a matrix converting the 4QD basis (of size 256) from computational to ((down, down), (S), (T_0), (up, up)) tensor product itself

    half_matrix = np.eye(16)
    half_matrix[6][6] = -1/np.sqrt(2)
    half_matrix[6][9] = 1/np.sqrt(2)
    half_matrix[9][6] = 1/np.sqrt(2)
    half_matrix[9][9] = 1/np.sqrt(2)

    full_matrix = np.kron(half_matrix, half_matrix)

    return full_matrix


def convert_44_256_vector(input_vector):
    # Converts between 44-basis (net half-occupations) and 256-basis (all occupations) vectors 

    if len(input_vector) not in [44, 256]:
        return "Error: length of input vector is not 44 or 246"
    
    dummy_fock_basis = create_bit_strings(8)
    half_basis_idxs = [dummy_fock_basis.tolist().index(half_basis.tolist()) for half_basis in four_hubbard_states] 

    if len(input_vector) == 44:
        output_vector = np.zeros(256, dtype=input_vector.dtype)
        for half_basis_idx, full_basis_idx in enumerate(half_basis_idxs):
            output_vector[full_basis_idx] = input_vector[half_basis_idx]
    elif len(input_vector) == 256:
        output_vector = np.zeros(44, dtype=input_vector.dtype)
        for half_basis_idx, full_basis_idx in enumerate(half_basis_idxs):
            output_vector[half_basis_idx] = input_vector[full_basis_idx]

    return output_vector
            
    
def get_eigen(matrix):
    ret_dict = {}
    evecs = []
    eig_res = np.linalg.eig(matrix)
    ret_dict['eigenvalues'] = eig_res.eigenvalues
    for k in range(len(eig_res.eigenvectors[0])):
        evecs.append(eig_res.eigenvectors[:,k])
    ret_dict['eigenvectors'] = evecs

    return ret_dict


def pseudospin_total(state):
    # Returns R^2 * |state>
    # R^2: total pseudospin operator for half-filled 4-site case

    idxs = [0, 1, 2, 3]
    ret_state = np.zeros(len(state))

    for i in idxs:
        for j in idxs:
            print(ret_state)
            ret_state = ret_state + 0.5 * (-1) ** (i + j) * (
                (creation(creation(annihilation(annihilation(state, i, 'u'), i, 'd'), j, 'd'), j, 'u')) + 
                (annihilation(annihilation(creation(creation(state, i, 'd'), i, 'u'), j, 'u'), j, 'd'))
            )

    return ret_state


def varying_tunnelling(time_arr, starting_t = 1):
    # returns an array of changing tunnelling constants over a time array
    # can just hardcode whatever function but testing (simple) oscillatory behaviour first

    amplitude = 0.5
    mean = 1
    omega = 1

    ret_tunnelling = amplitude * np.sin(omega * time_arr) + mean
    
    return ret_tunnelling


def hubbard_tdse_hamiltonian_for_solver(t, state):
    # This is the Hamiltonian to be used for odeintw to solve for the time-varying case
    # Note: we must multiply by (-i/hbar) to rearrange TDSE for the solver to work
    # Note: this is the 44x44 Hamiltonian

    hbar = 1.0546E-34
    q_e = 1.602E-19
    bohr_m = 9.274E-24

    dummy_fock_basis = create_bit_strings(8)
    half_basis_idxs = [dummy_fock_basis.tolist().index(half_basis.tolist()) for half_basis in four_hubbard_states]
    half_basis_idxs.extend(n for n in range(256) if n not in half_basis_idxs)

    U = ((1E-5) * 1.602E-19)

    # Time-control for tunnelling terms
    t_four = np.array([
        [0, 2 * U/10, 0, 0],
        [0, 0, 1 * U/10, 0],
        [0, 0, 0, 2 * U/10],
        [0, 0, 0, 0]
    ])

    # Time-control for applied B-field
    pauli_x = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    omega = 2.13E9  # corresponding to the energy transition between singlet and triplet
    B_ampl = 0
    B = B_ampl * bohr_m * np.cos(omega * t)
    sum_of_pauli_xs = (
        np.kron(pauli_x, np.eye(64)) + 
        np.kron(np.eye(4), np.kron(pauli_x, np.eye(16))) +
        np.kron(np.eye(16), np.kron(pauli_x, np.eye(4))) + 
        np.kron(np.eye(64), pauli_x)
    )

    hamiltonian_B_full_idxs = B * sum_of_pauli_xs
    hamiltonian_B = hamiltonian_B_full_idxs[half_basis_idxs, :]
    hamiltonian_B = hamiltonian_B[:, half_basis_idxs]
    hamiltonian_B = hamiltonian_B[:44, :44]

    # Time-control for pulse
    energy_transition = hbar * omega
    # voltage = energy_transition / q_e

    hamiltonian_pulse_full_idxs = energy_transition * sum_of_pauli_xs
    hamiltonian_pulse = hamiltonian_pulse_full_idxs[half_basis_idxs, :]
    hamiltonian_pulse = hamiltonian_pulse[:, half_basis_idxs]
    hamiltonian_pulse = hamiltonian_pulse[:44, :44]

    # Combine terms to get final Hamiltonian
    final_hamiltonian = (create_four_hubbard_hamiltonian_half_occ(t_four, U) + hamiltonian_B + hamiltonian_pulse) * (-1j / hbar)

    return np.dot(final_hamiltonian, state)


def S_T_pauli_X(basis=0):
    # Pauli X in Singlet-triplet (T_0) subspace
    # basis = 0: computational basis
    # basis = 1: S-T_0 basis

    S_T_pauli_X_S_T_basis = np.eye(16)
    S_T_pauli_X_S_T_basis[6][6] = 0
    S_T_pauli_X_S_T_basis[6][9] = 1
    S_T_pauli_X_S_T_basis[9][6] = 1
    S_T_pauli_X_S_T_basis[9][9] = 0

    if basis == 0:
        convert_bases = np.eye(16)
        convert_bases[6][6] = -1/np.sqrt(2)
        convert_bases[6][9] = 1/np.sqrt(2)
        convert_bases[9][6] = 1/np.sqrt(2)
        convert_bases[9][9] = 1/np.sqrt(2)
        S_T_pauli_X_comp_basis = np.dot(convert_bases.T, np.dot(S_T_pauli_X_S_T_basis, convert_bases))
        return S_T_pauli_X_comp_basis
    elif basis == 1:
        return S_T_pauli_X_S_T_basis
    else:
        return 'Error: basis must be 0 or 1'


# --- STATE PROBABILITY PLOTTING ---

def compute_prob(n, base_states_dummy, time_state):
    obs_state = base_states_dummy[n]
    return prob_of_state(obs_state, time_state)


def prob_plot_simple(basis=0):
    # basis = 0 for computational basis, basis = 1 for S-T_0 basis

    hbar = 1.0546E-34
    U = ((1E-5) * 1.602E-19)
    omega = 1E9  # corresponding to the energy transition between singlet and triplet
    t_four = np.array([
        [0, 1 * U/10, 0, 0],
        [0, 0, 0 * U/10, 0],
        [0, 0, 0, 1 * U/10],
        [0, 0, 0, 0]
    ])
    energy_transition = hbar * omega

    S_T_pauli_X_comp = S_T_pauli_X(basis=0)
    sum_of_S_T_pauli_X = np.kron(S_T_pauli_X_comp, np.eye(16)) + np.kron(np.eye(16), S_T_pauli_X_comp)
    pulse_hamiltonian = energy_transition * sum_of_S_T_pauli_X

    four_hubbard = create_four_hubbard_hamiltonian(t_four, U)
    four_hubbard = four_hubbard + pulse_hamiltonian
    four_hubbard_evol = -1j * four_hubbard / hbar
    T = np.linspace(0E-9, 5E-9, 1000)
    init_state = np.zeros(256)

    init_state[149] = 1/np.sqrt(2)
    init_state[101] = -1/np.sqrt(2)

    time_state = scipy.sparse.linalg.expm_multiply(four_hubbard_evol, init_state, T[0], T[-1], len(T), endpoint=True)

    convert_to_T_S_basis = computational_to_t_s_basis_change()
    if basis == 1:
        time_state = np.dot(time_state, convert_to_T_S_basis.T)
    elif basis == 0:
        pass
    else:
        return "Error: basis must be 0 or 1"
    
    dummy = create_bit_strings(8)
    base_states_dummy = generate_base_states(len(dummy))

    results = np.array([prob_of_state(base_state, time_state) for base_state in base_states_dummy])
    rounded_results = np.round(results, 3)
    cmap = plt.get_cmap('tab20')
    color_idx = 0
    minimum_prob = 0.01  # control plotting of low-probability states
    plt.figure(figsize=(40, 6))

    for n, res in enumerate(results):
        res = np.nan_to_num(res)
        rounded_res = np.nan_to_num(rounded_results[n])
        if rounded_res.any() and np.max(res) > minimum_prob:
            if basis == 0:
                idx = four_hubbard_states.tolist().index(dummy[n].tolist())
                label = '|' + ','.join(four_hubbard_states_arrows[idx]) + '>'
                plt.plot(T, res, label=label, color=cmap(color_idx % 20))
                color_idx += 1
            elif basis == 1:
                label = '|' + ','.join(four_T_S_states_arrows[n]) + '>'
                plt.plot(T, res, label=label, color=cmap(color_idx % 20))
                color_idx += 1
            else:
                return "Error: basis must be 0 or 1"

    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.show()


# def tdse_prob_plot(basis=1):
#     # Time dependent Schrodinger Equation solution and state probability plotting

#     init_state = np.zeros(256)

#     init_state[149] = 1/np.sqrt(2)
#     init_state[101] = -1/np.sqrt(2)
#     init_state = convert_44_256_vector(init_state)

#     T = np.linspace(0, 4E-9, 300)
#     T_span = (T[0], T[-1])

#     solution = solve_ivp(hubbard_tdse_hamiltonian_for_solver, T_span, init_state, t_eval=T, method='RK45')
#     proper_solution = []
#     for n in range(len(T)):
#         proper_solution.append(solution.y[:, n])
#     proper_solution = np.array(proper_solution)


# Still in development: adjusting tunnelling coefficients at some time
def prob_plot_varying_pulse(basis=1):
    # basis = 0: computational basis; basis = 1: S-T_0 basis

    hbar = 1.0546E-34
    U = ((1E-5) * 1.602E-19)
    omega = 1E9 

    tunnelling_1 = np.array([
        [0, 1 * U/10, 0, 0],
        [0, 0, 0 * U/10, 0],
        [0, 0, 0, 1 * U/10],
        [0, 0, 0, 0]
    ])
    tunnelling_2 = np.array([
        [0, 1 * U/10, 0, 0],
        [0, 0, 4 * U/10, 0],
        [0, 0, 0, 1 * U/10],
        [0, 0, 0, 0]
    ])
    hubbard_hamiltonian_1 = create_four_hubbard_hamiltonian(tunnelling_1, U)
    hubbard_hamiltonian_2 = create_four_hubbard_hamiltonian(tunnelling_2, U)

    energy_applied_1 = hbar * omega 
    energy_applied_2 = hbar * omega
    S_T_pauli_X_comp = S_T_pauli_X(basis=0)
    sum_of_S_T_pauli_X = np.kron(S_T_pauli_X_comp, np.eye(16)) + np.kron(np.eye(16), S_T_pauli_X_comp)
    pulse_hamiltonian_1 = energy_applied_1 * sum_of_S_T_pauli_X
    pulse_hamiltonian_2 = energy_applied_2 * sum_of_S_T_pauli_X

    evol_hamiltonian_1 = -1j * (hubbard_hamiltonian_1 + pulse_hamiltonian_1) / hbar
    evol_hamiltonian_2 = -1j * (hubbard_hamiltonian_2 + pulse_hamiltonian_2) / hbar

    time_1 = np.linspace(0, 1.5E-9, 500)
    time_2 = np.linspace(1.5E-9, 8E-9, 1500)
    time_all = np.concatenate((time_1, time_2[1::]), axis=0)
    time_2_adjusted = time_2 - time_2[0]

    init_state_1 = np.zeros(256)
    init_state_1[149] = 1/np.sqrt(2)
    init_state_1[101] = -1/np.sqrt(2)

    time_state_1 = scipy.sparse.linalg.expm_multiply(evol_hamiltonian_1, init_state_1, time_1[0], time_1[-1], len(time_1), endpoint=True)
    init_state_2 = time_state_1[-1]
    time_state_2 = scipy.sparse.linalg.expm_multiply(evol_hamiltonian_2, init_state_2, time_2_adjusted[0], time_2_adjusted[-1], len(time_2_adjusted), endpoint=True)
    time_state_all = np.concatenate((time_state_1, time_state_2[1::,:]), axis=0)

    convert_to_S_T_basis = computational_to_t_s_basis_change()
    if basis == 1:
        time_state_all = np.dot(time_state_all, convert_to_S_T_basis.T)
    elif basis == 0:
        pass
    else:
        return 'Error: basis must be 0 or 1'
    
    dummy = create_bit_strings(8)
    base_states_dummy = generate_base_states(len(dummy))

    results = np.array([prob_of_state(base_state, time_state_all) for base_state in base_states_dummy])
    rounded_results = np.round(results, 3)
    cmap = plt.get_cmap('tab20')
    color_idx = 0
    minimum_prob = 0.1  # control plotting of low-probability states
    plt.figure(figsize=(40, 6))

    for n, res in enumerate(results):
        res = np.nan_to_num(res)
        rounded_res = np.nan_to_num(rounded_results[n])
        if rounded_res.any() and np.max(res) > minimum_prob:
            if basis == 0:
                idx = four_hubbard_states.tolist().index(dummy[n].tolist())
                label = '|' + ','.join(four_hubbard_states_arrows[idx]) + '>'
                plt.plot(time_all, res, label=label, color=cmap(color_idx % 20))
                color_idx += 1
            elif basis == 1:
                label = '|' + ','.join(four_T_S_states_arrows[n]) + '>'
                plt.plot(time_all, res, label=label, color=cmap(color_idx % 20))
                color_idx += 1
            else:
                return 'Error: basis must be 0 or 1'
            
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.show()


# --- PURITY PLOTTING ---

def purity_plot():
    # Plots purity of substates against time
    # Purity = trace(partial trace over unwanted subspace(state density matrix) ^ 2)
    hbar = 1
    U = 10
    ratio = 0.01
    t_four = np.array([
        [0, U/10, 0, 0],
        [0, 0, np.sqrt(ratio*U), 0],
        [0, 0, 0, U/10],
        [0, 0, 0, 0]
    ])
    T = np.linspace(0, 300, 10000)
    subspaces = [[0, 1]]  # QDs to look at, 0-indexed
    # for i in [106]:
    for subspace in subspaces:
        four_hubbard = create_four_hubbard_hamiltonian(t_four, U)
        four_hubbard_evol = four_hubbard * (-1j) / hbar
        init_state = np.zeros(256)
        # init_state_idx = i
        init_state[153] = 1/2
        init_state[150] = -1/2
        init_state[105] = -1/2
        init_state[102] = 1/2
        time_state = scipy.sparse.linalg.expm_multiply(four_hubbard_evol, init_state, T[0], T[-1], len(T), endpoint=True)
        plt.figure(figsize=(40,6))
        purities = []
        for state in time_state:
            qstate = qt.Qobj(state, dims=[[4,4,4,4],[1,1,1,1]])
            rho = qstate * qstate.dag()
            rho_red = rho.ptrace(subspace)  # partial trace LEAVES subspace in i.e. tracing out {space}\{subspace}
            rho_red_sq = rho_red ** 2
            purity = rho_red_sq.tr()
            purity = np.round(purity, 4)
            purities.append(purity)
        # dummy = create_bit_strings(8)
        # idx = four_hubbard_states.tolist().index(dummy[init_state_idx].tolist())
        # init_state_str = '|' + ','.join(four_hubbard_states_arrows[idx]) + '>'
        # title = f'Purity of qubits {(np.array(subspace) + 1).tolist()}, initial state = {init_state_str}'
        plt.plot(T, purities)
        plt.xlabel('Time')
        plt.ylabel('Purity')
        # plt.title(title)
        plt.show()


# --- STATE PROBABILITY COLOURMAP (TUNNELLING AND TIME) PLOTTING ---

def prob_colourmap():
    # Colourmap of state probability with varying tunnelling (y-axis) and time (x-axis)
    hbar = 1
    t_four = np.array([
        [0, 0.8, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0.8],
        [0, 0, 0, 0]
    ])
    U = 7.6
    T = np.linspace(0, 50, 200)
    th_vals = np.linspace(0, 4, 25)
    control_barrier = 2

    idx_init = 153
    idx_obs = 153

    init_state = np.zeros(256)
    obs_state = np.zeros(256)

    init_state[idx_init] = 1
    obs_state[idx_obs] = 1

    # label_init = '|' + ','.join(four_hubbard_states_arrows[idx_init]) + '>'
    # label_obs = '|' + ','.join(four_hubbard_states_arrows[idx_obs]) + '>'

    def hubbard_vary_tunnelling_time(th, T):
        four_hubbard = create_four_hubbard_hamiltonian(th, U)
        H = (-1j) * four_hubbard / hbar
        time_states = scipy.sparse.linalg.expm_multiply(H, init_state, T[0], T[-1], len(T), endpoint=True)
        ret_probs = prob_of_state(obs_state, time_states)

        return ret_probs

    colourmap_vals = np.zeros((len(th_vals), len(T)))

    for i, th in enumerate(th_vals):
        t_four[control_barrier - 1][control_barrier] = th
        colourmap_vals[i,:] = hubbard_vary_tunnelling_time(t_four, T)

    plt.figure(figsize=(90, 30))
    plt.imshow(colourmap_vals, 
            aspect='auto', 
            extent=[T[0], T[-1], th_vals[0], th_vals[-1]], 
            origin='lower', 
            cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel(f'Tunnelling parameter (qubit {control_barrier} to qubit {control_barrier + 1})')
    # plt.title(f'Probabilities of {label_obs} (initial state = {label_init}, U {U})')
    plt.show()


# --- PARTIAL TRACE COLOURMAP (TUNNELLING AND TIME) PLOTTING ---

def purity_colourmap():
    # Colourmap of subspace purity with varying tunnelling coefficient (y-axis) and time (x-axis)
    hbar = 1
    t_four = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    # t_sq_by_U = 2.02766  # found "optimal" t^2/U ratio
    # U_arr = np.linspace(0.001, 20, 1000)
    T = np.linspace(0, 300, 10500)
    ratio_arr = np.linspace(1.5, 2.5, 101)
    U = 15

    barrier = 1  # potential barrier between qubit (barrier) and qubit (barrier + 1) being controlled, 0-indexed
    subspace = [0,1]  # taking partial trace over {space}\{subspace}, 0-indexed

    idx_init = 102
    init_state = np.zeros(256)
    init_state[idx_init] = 1

    def subspace_purities_over_time(states, subspace):
        # returns an array of purity values of partial traces over states (an array of statevectors over time), keeping (subspace) qubits

        purities = []
        for state in states:
            qstate = qt.Qobj(state, dims=[[4,4,4,4], [1,1,1,1]])
            rho = qstate * qstate.dag()
            rho_red = rho.ptrace(subspace)
            rho_red_sq = rho_red ** 2
            purity = rho_red_sq.tr()
            purities.append(purity)

        return np.array(purities)

    def ptrace_vary_tunnelling_time(tunnelling_params, T, U_C):
        H_four = create_four_hubbard_hamiltonian(tunnelling_params, U_C)
        H = (-1j) * H_four / hbar
        time_states = scipy.sparse.linalg.expm_multiply(H, init_state, T[0], T[-1], len(T), endpoint=True)
        purities = subspace_purities_over_time(time_states, subspace)

        return purities

    colourmap_vals = np.zeros((len(ratio_arr), len(T)))

    for i, ratio in enumerate(ratio_arr):
        t_four[barrier][barrier + 1] = np.sqrt(ratio * U)
        colourmap_vals[i,:] = ptrace_vary_tunnelling_time(t_four, T, U)

    np.save('purity_colourmap_time_0_300_ratio_1.5_2.5_U_15_t0_1_init_102', colourmap_vals)

    # dummy = create_bit_strings(8)
    # idx = four_hubbard_states.tolist().index(dummy[idx_init].tolist())
    # init_state_str = '|' + ','.join(four_hubbard_states_arrows[idx]) + '>'
    # plt.figure(figsize=(90, 30))
    # plt.imshow(colourmap_vals,
    #         aspect='auto',
    #         extent=[T[0], T[-1], th_sq_by_U[0], th_sq_by_U[-1]],
    #         origin='lower',
    #         cmap='viridis')
    # plt.xlabel('Time')
    # plt.ylabel('$t^2_{23}$ / U')
    # plt.title(f'Purity of qubits {(np.array(subspace) + 1).tolist()}, initial state = {init_state_str}')
    # plt.colorbar()
    # plt.show()
    
    
# --- ATTEMPTS AT ANALYTICAL DIAGONALISATION (still incomplete) ---

def symbolic_hubbard_hamiltonian():
    # Returns a sympy.Matrix symbolic 4-site Hubbard Hamiltonian (in the 44 dimensional Hilbert subspace)
    # Arrangement in the order defined by the basis states in Hubbard_utils.py
    
    U_coulomb = 10
    t_four = np.array([
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0]
    ])
    t12, t23, t34, U = sp.symbols('t12,t23,t34,U')
    num_to_sym_mapping = {1.0: t12, -1.0: -t12, 2.0: t23, -2.0: -t23, 3.0: t34, 
                          -3.0: -t34, 10.0: U, -10.0: -U, 20.0: 2*U, -20.0: -2*U}  # arbitrary numbers to avoid typing out 44x44 matrix
    
    temp_num_hamiltonian = create_four_hubbard_hamiltonian_half_occ(t_four, U_coulomb)
    def num_to_sym_replace(value):
        return num_to_sym_mapping.get(value, value)
    
    symbolic_hubbard = [[num_to_sym_replace(value) for value in row] for row in temp_num_hamiltonian]
    symbolic_hubbard = sp.Matrix(symbolic_hubbard)
    
    return symbolic_hubbard
    

def Sz_separation():
    # Returns sympy.Matrix symbolic 4-site Hubbard Hamiltonian (in 44 dimensional Hilbert subspace)
    # Arrangement in the order of Sz = 0, +-1, +-2 (i.e. block diagonalised), mapping in Hubbard_utils.py
    
    sym_hamiltonian = symbolic_hubbard_hamiltonian()

    rearr_dummy = []
    for new_idx, old_idx in rearr_Sz_subspace:
        rearr_dummy.append(sym_hamiltonian.row(old_idx))
    rearr_sym_hamiltonian = sp.Matrix(rearr_dummy)
    
    rearr_dummy = []
    for new_idx, old_idx in rearr_Sz_subspace:
        rearr_dummy.append(rearr_sym_hamiltonian.col(old_idx))
    rearr_sym_hamiltonian = sp.Matrix.hstack(*rearr_dummy)
    
    return rearr_sym_hamiltonian
