#include "utils_backup.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif


void eigen_matrix_exponential(const ComplexeMatrix* A, ComplexeMatrix* out);

#ifdef __cplusplus
}
#endif


#define MAX_TERMS 100
#define TOL 1e-12
#define HBAR 1.054571817e-34



int find_state_index(StateList* basis, State* init_state, int size);

void print_state(const State *s) {
    printf("État (taille = %d) : ", s->size);
    for (int i = 0; i < s->size; i++) {
        printf("%d ", s->occupancy[i]);
    }
    printf("\n");
}

void print_statelist(const StateList* state_list, int show_details) {
    if (!state_list) {
        printf("StateList: NULL\n");
        return;
    }
    printf("=== StateList ===\n");
    printf("Nombre d'états: %d\n", state_list->count);
    if (!state_list->states) {
        printf("États: NULL (liste vide)\n");
        return;
    }
    if (state_list->count == 0) {
        printf("États: Liste vide\n");
        return;
    }
    if (show_details) {
        printf("\nDétails des états:\n");
        for (int i = 0; i < state_list->count; i++) {
            printf("  État %d: ", i);
            print_state(&state_list->states[i]);
        }
    } else {
        printf("États: %d éléments présents\n", state_list->count);
        if (state_list->count > 0) {
            printf("Premier état: ");
            print_state(&state_list->states[0]);
            if (state_list->count > 1) {
                printf("Dernier état: ");
                print_state(&state_list->states[state_list->count - 1]);
            }
        }
    }
    printf("================\n\n");
}

StateComplexe* basis_vector(int dim, int idx) {
    StateComplexe* sc = malloc(sizeof(StateComplexe));
    sc->size = dim;
    sc->vector = calloc(dim, sizeof(cplx));
    if (idx < 0 || idx >= dim) {
        fprintf(stderr, "Erreur: idx=%d hors limites [0, %d)\n", idx, dim);
        return sc;
    }
    sc->vector[idx] = 1.0 + 0.0*I;
    return sc;
}

Matrix* allocate_memory_matrix(int dim) {
    Matrix *H = malloc(sizeof *H);
    H->dim    = dim;
    H->matrix = malloc(dim * sizeof *H->matrix);
    for (int i = 0; i < dim; i++) {
        H->matrix[i] = calloc(dim, sizeof *H->matrix[i]);
    }
    return H;
}

void free_memory_matrix(Matrix *H) {
    if (!H) return;
    for (int i = 0; i < H->dim; i++)
        free(H->matrix[i]);
    free(H->matrix);
    free(H);
}

// StateComplexe* basis_vector(int dim, int idx) {
//     StateComplexe* sc = malloc(sizeof(StateComplexe));
//     sc->size = dim;
//     sc->vector = calloc(dim, sizeof(cplx));
//     sc->vector[idx] = 1.0 + 0.0*I;
//     return sc;
// }

static int binomial_coefficient(int n, int k) {
    if (n < 0 || k < 0) return -1;
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    int res = 1;
    for (int i = 1; i <= k; i++)
        res = res * (n - k + i) / i;
    return res;
}

CombinationList* init_combination_list(int k, int n) {
    CombinationList *L = malloc(sizeof *L);
    L->count     = 0;
    L->max_count = binomial_coefficient(n, k);

    // PATCH: Vérifie qu’on ne va pas allouer de la folie
    if (L->max_count <= 0 || L->max_count > 10000000) {
        fprintf(stderr, "Impossible d’allouer %lld combinaisons pour n=%d, k=%d !\n",
                (long long)L->max_count, n, k);
        free(L);
        return NULL;
    }

    L->combinations = malloc(L->max_count * sizeof (Combination));
    for (int i = 0; i < L->max_count; i++) {
        L->combinations[i].size    = k;
        L->combinations[i].indices = malloc(k * sizeof *L->combinations[i].indices);
    }
    return L;
}

CombinationList* combinations_iterative(int k, int n) {
    if (k > n || k < 0) return NULL;
    CombinationList *L = init_combination_list(k, n);
    int *comb = malloc(k * sizeof *comb);
    for (int i = 0; i < k; i++) comb[i] = i;
    do {
        memcpy(L->combinations[L->count].indices, comb, k * sizeof *comb);
        L->count++;
        int pos = k - 1;
        while (pos >= 0 && comb[pos] == n - k + pos) pos--;
        if (pos < 0) break;
        comb[pos]++;
        for (int j = pos + 1; j < k; j++)
            comb[j] = comb[j-1] + 1;
    } while (1);
    free(comb);
    return L;
}

void free_combination_list(CombinationList *L) {
    if (!L) return;
    for (int i = 0; i < L->max_count; i++)
        free(L->combinations[i].indices);
    free(L->combinations);
    free(L);
}

bool any(const State *s) {
    for (int i = 0; i < s->size; i++)
        if (s->occupancy[i] != 0) return 1;
    return 0;
}

State* make_zero_state(int dim) {
    State *Z = malloc(sizeof *Z);
    Z->size      = dim;
    Z->occupancy = calloc(dim, sizeof *Z->occupancy);
    return Z;
}

State* abs_state(const State *src) {
    State *A = malloc(sizeof *A);
    A->size      = src->size;
    A->occupancy = malloc(src->size * sizeof *A->occupancy);
    for (int i = 0; i < src->size; i++)
        A->occupancy[i] = llabs(src->occupancy[i]);
    return A;
}

bool state_equal(const State *a, const State *b) {
    if (a->size != b->size) return 0;
    for (int i = 0; i < a->size; i++)
        if (a->occupancy[i] != b->occupancy[i]) return 0;
    return 1;
}

int number_operator(const State *s, int site, char spin) {
    return spin=='u'
           ?  (int)s->occupancy[2*site]
           :  (int)s->occupancy[2*site+1];
}

State* annihilation(const State *st, int site, char spin) {
    int idx = (spin=='u' ? 2*site : 2*site+1);
    int sum = 0;
    for (int i = 0; i < idx; i++) sum += st->occupancy[i];
    int sign = (sum % 2 == 0) ? +1 : -1;

    if (!any(st) || st->occupancy[idx] == 0)
        return make_zero_state(st->size);

    State *out = malloc(sizeof *out);
    out->size      = st->size;
    out->occupancy = malloc(st->size * sizeof *out->occupancy);
    memcpy(out->occupancy, st->occupancy, st->size * sizeof *out->occupancy);
    out->occupancy[idx] = 0;
    for (int i = 0; i < out->size; i++)
        out->occupancy[i] *= sign;
    return out;
}

State* creation(const State *st, int site, char spin) {
    int idx = (spin=='u' ? 2*site : 2*site+1);
    int sum = 0;
    for (int i = 0; i < idx; i++) sum += st->occupancy[i];
    int sign = (sum % 2 == 0) ? +1 : -1;

    if (!any(st) || st->occupancy[idx] == 1)
        return make_zero_state(st->size);

    State *out = malloc(sizeof *out);
    out->size      = st->size;
    out->occupancy = malloc(st->size * sizeof *out->occupancy);
    memcpy(out->occupancy, st->occupancy, st->size * sizeof *out->occupancy);
    out->occupancy[idx] = 1;
    for (int i = 0; i < out->size; i++)
        out->occupancy[i] *= sign;
    return out;
}

double complex hermitian_dot(const double complex* a, const double complex* b, int size) {
    double complex result = 0.0 + 0.0 * I;
    for (int i = 0; i < size; i++)
        result += conj(a[i]) * b[i];
    return result;
}



double* generate_time_array(double T_final, int nbr_pts) {
    double* T_array = malloc(nbr_pts * sizeof(double));
    double dt = T_final / (nbr_pts - 1);
    for (int i = 0; i < nbr_pts; i++) T_array[i] = (i * dt); // / HBAR
    return T_array;
}

// double* get_sampling_timestep(Matrix* H){

// }

StateComplexe* convert_state_to_hilbert_vector(State* s, StateList* basis) {
    // Trouver l'index de l'état s dans la base
    int idx = -1;
    for (int i = 0; i < basis->count; i++) {
        if (state_equal(s, &basis->states[i])) {
            idx = i;
            break;
        }
    }
    
    if (idx == -1) {
        fprintf(stderr, "État non trouvé dans la base!\n");
        return NULL;
    }
    
    // Créer un vecteur de base dans l'espace de Hilbert
    return basis_vector(basis->count, idx);
}

// 2. FIXED: Time evolution operator with better error handling
ComplexeMatrix* time_evol_operator(Matrix* H, double t) {

    
    if (!H || !isfinite(t) || t < 0) {
        fprintf(stderr, "Erreur: paramètres invalides dans time_evol_operator\n");
        return NULL;
    }
    
    // Create -i*H*t/ℏ
    ComplexeMatrix* A = allocate_complex_matrix(H->dim);
    if (!A) return NULL;
    
    for (int i = 0; i < H->dim; i++) {
        for (int j = 0; j < H->dim; j++) {
            double element = -H->matrix[i][j] * t / HBAR;
            A->compMatrix[i][j] = 0.0 + element * I;  // Pure imaginary
        }
    }
    
    // Check if matrix elements are reasonable
    double max_element = 0.0;
    for (int i = 0; i < A->dim; i++) {
        for (int j = 0; j < A->dim; j++) {
            double mag = cabs(A->compMatrix[i][j]);
            if (mag > max_element) max_element = mag;
        }
    }
    
    if (max_element > 50.0) {  // Prevent numerical overflow
        //printf("Avertissement: éléments de matrice très grands (%.2e) à t=%.2e\n", max_element, t);
    }
    
    ComplexeMatrix* U = matrix_exponential_fixed(A);
    free_complex_matrix(A);
    
    return U;
}

// 4. FIXED: Time evolution with numerical stability checks
StateListComplexe* time_evol_state_enhanced(Matrix* H, double* T_array, int nbr_pts, StateComplexe* u) {
    if (!H || !T_array || !u || nbr_pts <= 0) {
        fprintf(stderr, "Erreur: paramètres invalides\n");
        return NULL;
    }
    
    StateListComplexe* result_array = malloc(sizeof(StateListComplexe));
    if (!result_array) {
        fprintf(stderr, "Erreur: allocation mémoire échouée\n");
        return NULL;
    }
    
    result_array->count = nbr_pts;
    result_array->complexe_state = malloc(nbr_pts * sizeof(StateComplexe));
    if (!result_array->complexe_state) {
        free(result_array);
        return NULL;
    }

    // Initialisation des vecteurs
    for (int i = 0; i < nbr_pts; i++) {
        result_array->complexe_state[i].size = u->size;
        result_array->complexe_state[i].vector = calloc(u->size, sizeof(cplx));
        if (!result_array->complexe_state[i].vector) {
            for (int k = 0; k < i; k++) free(result_array->complexe_state[k].vector);
            free(result_array->complexe_state);
            free(result_array);
            fprintf(stderr, "Erreur: allocation état %d échouée\n", i);
            return NULL;
        }
    }

    // état initial
    memcpy(result_array->complexe_state[0].vector, u->vector, u->size * sizeof(cplx));

    // configuration de la barre
    const int barWidth = 50;
    printf("Évolution temporelle :\n");

    // boucle de calcul
    for (int t_idx = 1; t_idx < nbr_pts; t_idx++) {
        double t = T_array[t_idx];
        if (!isfinite(t) || t < 0) {
            memcpy(result_array->complexe_state[t_idx].vector,
                   result_array->complexe_state[t_idx-1].vector,
                   u->size * sizeof(cplx));
            continue;
        }

        ComplexeMatrix* U = time_evol_operator(H, t);
        if (!U) {
            memcpy(result_array->complexe_state[t_idx].vector,
                   result_array->complexe_state[t_idx-1].vector,
                   u->size * sizeof(cplx));
            continue;
        }

        // application de U
        for (int i = 0; i < u->size; i++) {
            result_array->complexe_state[t_idx].vector[i] = 0.0 + 0.0*I;
            for (int j = 0; j < u->size; j++)
                result_array->complexe_state[t_idx].vector[i] += U->compMatrix[i][j] * u->vector[j];
        }
        free_complex_matrix(U);

        // affichage de la barre de progression
        float progress = (float)t_idx / (nbr_pts - 1);
        int pos = (int)(barWidth * progress);
        printf("\r[");
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos)   printf("=");
            else if (i == pos) printf(">");
            else           printf(" ");
        }
        printf("] %3d%%", (int)(progress * 100));
        fflush(stdout);

        // petit délai si vous voulez voir la barre défiler (optionnel)
        // usleep(1000);
    }

    printf("\nTerminé (%d points).\n", nbr_pts);
    return result_array;
}
StateListComplexe* time_evol_state(Matrix* H, double* T_array, int nbr_pts, StateComplexe* u) {
    if (!H || !T_array || !u || nbr_pts <= 0) {
        fprintf(stderr, "Erreur: paramètres invalides\n");
        return NULL;
    }

    StateListComplexe* result_array = malloc(sizeof(StateListComplexe));
    if (!result_array) {
        fprintf(stderr, "Erreur: allocation mémoire échouée\n");
        return NULL;
    }
    
    result_array->count = nbr_pts;
    result_array->complexe_state = malloc(nbr_pts * sizeof(StateComplexe));

    // Initialize all states
    for (int i = 0; i < nbr_pts; i++) {
        result_array->complexe_state[i].size = u->size;
        result_array->complexe_state[i].vector = calloc(u->size, sizeof(cplx));
        if (!result_array->complexe_state[i].vector) {
            fprintf(stderr, "Erreur: allocation mémoire échouée\n");
            for (int k = 0; k < i; k++) {
                free(result_array->complexe_state[k].vector);
            }
            free(result_array->complexe_state);
            free(result_array);
            return NULL;
        }
    }

    // First state is the initial state
    for (int i = 0; i < u->size; i++) {
        result_array->complexe_state[0].vector[i] = u->vector[i];
    }

    for (int t_idx = 1; t_idx < nbr_pts; t_idx++) {  // Start from 1, not 0
        double t = T_array[t_idx];
        
        // Check for reasonable time values
        if (t < 0 || !isfinite(t)) {
            fprintf(stderr, "Temps invalide: t[%d] = %e\n", t_idx, t);
            continue;
        }
        
        ComplexeMatrix* U = time_evol_operator(H, t);
        if (U == NULL) {
            fprintf(stderr, "Erreur: time_evol_operator a retourné NULL à t=%e\n", t);
            continue;
        }
        
        // Apply time evolution operator
        for (int i = 0; i < U->dim; i++) {
            result_array->complexe_state[t_idx].vector[i] = 0.0 + 0.0*I;
            for (int j = 0; j < U->dim; j++) {
                result_array->complexe_state[t_idx].vector[i] += 
                    U->compMatrix[i][j] * u->vector[j];
            }
        }
        
        // Check for numerical stability
        double norm = 0.0;
        for (int i = 0; i < u->size; i++) {
            double magnitude = cabs(result_array->complexe_state[t_idx].vector[i]);
            // if (!isfinite(magnitude)) {
            //     fprintf(stderr, "NaN détecté à t_idx=%d, i=%d\n", t_idx, i);
            //     break;
            // }
            norm += magnitude * magnitude;
        }
        
        // Normalize if needed (optional, but helps with stability)
        if (norm > 0 && isfinite(norm)) {
            norm = sqrt(norm);
            if (norm > 1.1 || norm < 0.9) {  // Renormalize if drift is significant
                for (int i = 0; i < u->size; i++) {
                    result_array->complexe_state[t_idx].vector[i] /= norm;
                }
            }
        }
        
        free_complex_matrix(U);
        
        if ((t_idx + 1) % (nbr_pts / 10) == 0 && nbr_pts >= 10) {
            printf("Progression: %d%%\n", (int)((t_idx + 1) * 100.0 / nbr_pts));
        }
    }
    
    printf("time_evol_state finie\n");
    return result_array;
}

StateListComplexe* time_evol_state_diag(Matrix* H, double* T_array, int nbr_pts, StateComplexe* u) {
    int dim = H->dim;
    // Prépare la matrice complexe à partir de la matrice réelle H
    ComplexeMatrix* Hc = allocate_complex_matrix(dim);
    for (int i=0; i<dim; ++i)
        for (int j=0; j<dim; ++j)
            Hc->compMatrix[i][j] = H->matrix[i][j] + 0.0*I;

    // Appel à la routine Eigen
    cplx* psi_out = malloc(nbr_pts * dim * sizeof(cplx));
    time_evolution_diagonalization(Hc, u->vector, dim, T_array, nbr_pts, psi_out);

    // Emballe le résultat dans StateListComplexe
    StateListComplexe* result = malloc(sizeof(StateListComplexe));
    result->count = nbr_pts;
    result->complexe_state = malloc(nbr_pts * sizeof(StateComplexe));
    for (int t = 0; t < nbr_pts; t++) {
        result->complexe_state[t].size = dim;
        result->complexe_state[t].vector = malloc(dim * sizeof(cplx));
        for (int i = 0; i < dim; i++)
            result->complexe_state[t].vector[i] = psi_out[t*dim + i];
    }

    free_complex_matrix(Hc);
    free(psi_out);
    return result;
}

StateList* get_hubbard_states(int N) {
    int dim = 2LL * N;
    CombinationList *C = combinations_iterative(N, dim);
    if (!C) return NULL;
    StateList *L = malloc(sizeof *L);
    L->count  = C->count;
    L->states = malloc(L->count * sizeof *L->states);
    for (int i = 0; i < C->count; i++) {
        L->states[i].size      = dim;
        L->states[i].occupancy = calloc(dim, sizeof *L->states[i].occupancy);
        for (int k = 0; k < C->combinations[i].size; k++) {
            int idx = C->combinations[i].indices[k];
            L->states[i].occupancy[idx] = 1;
        }
    }
    free_combination_list(C);
    return L;
}

void free_state_list(StateList *L) {
    if (!L) return;
    for (int i = 0; i < L->count; i++)
        free(L->states[i].occupancy);
    free(L->states);
    free(L);
}

Matrix* initialize_matrix_with_zeros(int dim) {
    return allocate_memory_matrix(dim);
}

void print_matrix(const Matrix *M) {
    for (int i = 0; i < M->dim; i++) {
        for (int j = 0; j < M->dim; j++) {
            printf("%10.8f ", M->matrix[i][j]);
        }
        printf("\n");
    }
}

int hopping_term_sign_factor(const State *state_i, int i, int k, char spin) {
    int idx_i = (spin=='u' ? 2*i   : 2*i+1);
    int idx_k = (spin=='u' ? 2*k   : 2*k+1);
    int min_idx = idx_i < idx_k ? idx_i : idx_k;
    int max_idx = idx_i < idx_k ? idx_k : idx_i;
    int S = 0;
    for (int j = min_idx+1; j < max_idx; j++)
        S += state_i->occupancy[j];
    return (S % 2) ? -1 : +1;
}

Matrix* create_tridiagonal_matrix(int N) {
    Matrix *M = allocate_memory_matrix(N);
    if (!M) {
        perror("Erreur d’allocation");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; i++) {
        if (i > 0)     M->matrix[i][i-1] = 1.0;
        if (i < N-1)   M->matrix[i][i+1] = 1.0;
    }
    return M;
}

Matrix* hubbard_hamiltonian_matrix(int N, Matrix* t_matrix, double U){
    StateList* statelist = get_hubbard_states(N);
    if (!statelist) {
        free_state_list(statelist);
        return NULL;
    }
    int hilbert_dim = (int)(statelist->count);
    Matrix* H = initialize_matrix_with_zeros(hilbert_dim);
    for(int i = 0; i < hilbert_dim; i++){
        State* state_i = &statelist->states[i];
        for(int j = 0; j < hilbert_dim; j++){
            State* state_j = &statelist->states[j];
            if(i == j){
                for (int site =0; site < N; site ++){
                    int n_up = number_operator(state_i, site, 'u');
                    int n_down = number_operator(state_i, site, 'd');
                    H->matrix[i][j] += U * n_up * n_down;
                }
            } else{
                for(int site1 = 0; site1 < N; site1++){
                    char spins[2] = {'u', 'd'};
                    for (int l = 0; l < 2; l++) {
                        char spin = spins[l];
                        State* temp = annihilation(state_i, site1, spin);
                        if(any(temp)){
                            int site2_list[2] = {site1-1, site1+1};
                            for(int s = 0; s < 2; s++){
                                if(0 <= site2_list[s] && site2_list[s] < N){
                                    int site2 = site2_list[s];
                                    State* final = creation(temp, site2, spin);
                                    if(state_equal(abs_state(final), state_j)){
                                        int sign = hopping_term_sign_factor(state_i, site1, site2, spin);
                                        H->matrix[i][j] -= t_matrix->matrix[site1][site2] * sign;
                                    }
                                    free(final->occupancy);
                                    free(final);
                                }
                            }
                        }
                        free(temp->occupancy);
                        free(temp);
                    }
                }
            }
        }
    }
    free_state_list(statelist);
    return H;
}

StateComplexe* convert_state_to_complex(State* s) {
    StateComplexe* sc = malloc(sizeof(StateComplexe));
    sc->size = s->size;
    sc->vector = malloc(sc->size * sizeof(cplx));
    for (int i = 0; i < sc->size; i++)
        sc->vector[i] = (cplx)(s->occupancy[i]) + 0.0 * I;
    return sc;
}

void free_state_complexe(StateComplexe* sc) {
    if (sc) {
        free(sc->vector);
        free(sc);
    }
}

void print_state_list_complexe(StateListComplexe* list) {
    if (!list) {
        printf("Liste vide (NULL).\n");
        return;
    }
    for (int i = 0; i < list->count; i++) {
        StateComplexe state = list->complexe_state[i];
        printf("État #%d : [", i);
        for (int j = 0; j < state.size; j++) {
            double complex z = state.vector[j];
            printf(" %.3f%+.3fi ", creal(z), cimag(z));
            if (j < state.size - 1) printf("|");
        }
        printf("]\n");
    }
}

void print_state_complexe(StateComplexe* state) {
    if (!state || !state->vector) {
        printf("État complexe vide ou NULL.\n");
        return;
    }
    printf("StateComplexe (taille = %d):\n[", state->size);
    for (int i = 0; i < state->size; i++) {
        double complex z = state->vector[i];
        printf(" %.3f%+.3fi ", creal(z), cimag(z));
        if (i < state->size - 1) printf("|");
    }
    printf("]\n");
}

double* transition_probability_over_time(StateComplexe* left_state, StateListComplexe* list) {
    // Alloue le tableau résultat
    double* probabilities = malloc(list->count * sizeof(double));
    if (!probabilities) return NULL;

    for (int t_idx = 0; t_idx < list->count; t_idx++) {
        cplx inner_product = 0.0 + 0.0 * I;
        StateComplexe* right_state = &list->complexe_state[t_idx];
        if (right_state->size != left_state->size) {
            fprintf(stderr, "Erreur : tailles d'états incompatibles à t = %d\n", t_idx);
            probabilities[t_idx] = 0.0;
            continue;
        }
        for (int i = 0; i < left_state->size; i++) {
            inner_product += conj(left_state->vector[i]) * right_state->vector[i];
        }
        probabilities[t_idx] = pow(cabs(inner_product), 2);
    }
    return probabilities;
}

void print_double_array(const double* arr, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%.18f", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

// 5. IMPROVED: Save function with better error handling
void save_top_hubbard_states_to_csv(
        StateListComplexe* psi_t,
        StateList* state_list_init,
        const char* filename)
{
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Impossible d'ouvrir %s\n", filename);
        return;
    }

    // Header
    fprintf(f, "t_idx,idx,proba,real,imag,occupation\n");

    for (int t = 0; t < psi_t->count; t++) {
        for (int k = 0; k < psi_t->complexe_state[t].size; k++) {
            double complex coeff = psi_t->complexe_state[t].vector[k];
            double amp = cabs(coeff);
            double prob = amp * amp;
            double re = creal(coeff);
            double im = cimag(coeff);

            // Check for valid values before writing
            if (isfinite(prob) && isfinite(re) && isfinite(im)) {
                fprintf(f, "%d,%d,%.12e,%.12e,%.12e,[", t, k, prob, re, im);
            } else {
                fprintf(f, "%d,%d,0.0,0.0,0.0,[", t, k);
            }
            
            // Write occupation
            if (k < state_list_init->count) {
                for (int s = 0; s < state_list_init->states[k].size; s++) {
                    fprintf(f, "%d", state_list_init->states[k].occupancy[s]);
                    if (s < state_list_init->states[k].size - 1)
                        fprintf(f, " ");
                }
            }
            fprintf(f, "]\n");
        }
    }
    fclose(f);
}

void top_hubbard_states_interface(
    int     N,
    double  U,
    double  T_final,
    int     nbr_pts,
    State  *init_state,
    const char *filename,
    double  t_hopping
) {
    // Build hopping matrix and Hamiltonian
    Matrix* tmat = create_tridiagonal_matrix(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (tmat->matrix[i][j] != 0.0) {
                tmat->matrix[i][j] = t_hopping;
            }
        }
    }
    Matrix* H = hubbard_hamiltonian_matrix(N, tmat, U);

    // Generate time evolution
    double* T_array = generate_time_array(T_final, nbr_pts);
    StateList* state_list = get_hubbard_states(N);
    int init_idx = find_state_index(state_list, init_state, init_state->size);
    StateComplexe* v0 = basis_vector(state_list->count, init_idx);
    StateListComplexe* psi_t = time_evol_state_diag(H, T_array, nbr_pts, v0);

    // Save results to CSV and return immediately
    save_top_hubbard_states_to_csv(psi_t, state_list, filename);
    return;
}

int find_state_index(StateList* basis, State* init_state, int size) {
    for (int i = 0; i < basis->count; i++) {
        int match = 1;
        for (int j = 0; j < size; j++) {
            if (basis->states[i].occupancy[j] != init_state->occupancy[j]) {
                match = 0;
                break;
            }
        }
        if (match) return i;
    }
    return -1;
}

void print_matrix_complex(ComplexeMatrix* m) {
    if (m == NULL) {
        printf("Matrice NULL\n");
        return;
    }

    printf("Matrice %dx%d :\n", m->dim, m->dim);
    for (int i = 0; i < m->dim; i++) {
        for (int j = 0; j < m->dim; j++) {
            double real = creal(m->compMatrix[i][j]);
            double imag = cimag(m->compMatrix[i][j]);
            if (imag >= 0)
                printf(" %.2f+%.2fi ", real, imag);
            else
                printf(" %.2f%.2fi ", real, imag);
        }
        printf("\n");
    }
    printf("\n");
}