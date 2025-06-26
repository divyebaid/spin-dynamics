#ifndef UTILS_BACKUP_H
#define UTILS_BACKUP_H

#ifdef __cplusplus
    #include <complex>
    typedef std::complex<double> cplx;
    extern "C" {
#else
    #include <complex.h>
    #include <stdbool.h>
    typedef double complex cplx;
#endif

/* --- États quantiques --- */
typedef struct {
    int    size;       // nombre de cases dans occupancy
    int   *occupancy;  // tableau d’occupation 0/1
} State;

typedef struct {
    int size;       // Taille de l'état
    cplx* vector;   // Tableau de coefficients complexes
} StateComplexe;

typedef struct {
    State      *states;
    int   count;
} StateList;

typedef struct {
    StateComplexe* complexe_state;
    int count;
} StateListComplexe;

/* --- Combinaisons pour get_hubbard_states --- */
typedef struct {
    int *indices;
    int  size;
} Combination;

typedef struct {
    Combination *combinations;
    int    count;
    int    max_count;
} CombinationList;

/* --- Matrices --- */
typedef struct {
    double    **matrix;
    int   dim;
} Matrix;

typedef struct {
    int dim;
    cplx **compMatrix;
} ComplexeMatrix;

/* --- Prototypes --- */
void            print_state(const State *s);
void            print_state_list(const StateList *L);
void            print_matrix(const Matrix *M);

Matrix*         allocate_memory_matrix(int dim);
Matrix*         initialize_matrix_with_zeros(int dim);
void            free_memory_matrix(Matrix *M);

int             hopping_term_sign_factor(const State *state_i, int i, int k, char spin);

void            print_state_complexe(StateComplexe* state);

CombinationList* combinations_iterative(int k, int n);
void            free_combination_list(CombinationList *L);

StateList*      get_hubbard_states(int N);
void            free_state_list(StateList *L);
void            print_state_list_complexe(StateListComplexe* list);

ComplexeMatrix* time_evol_operator(Matrix* H, double t);
StateListComplexe* time_evol_state(Matrix* H, double* T_array, int nbr_pts, StateComplexe* u);
double*         transition_probability_over_time(StateComplexe* left_state, StateListComplexe * list);

int             number_operator(const State *s, int site, char spin);
State*          annihilation(const State *s, int site, char spin);
State*          creation    (const State *s, int site, char spin);

Matrix*         hubbard_hamiltonian_matrix(int N, Matrix *tmat, double U);
void            top_hubbard_states_interface(int N, double U, double T_final, int nbr_pts, State* init_state, const char* filename, double t_hopping);

/* --- Eigen-based exponentielle de matrice complexe --- */
ComplexeMatrix* allocate_complex_matrix(int dim);
void free_complex_matrix(ComplexeMatrix* matrix);
ComplexeMatrix* mat_identity(int dim);
ComplexeMatrix* mat_add(ComplexeMatrix* a, ComplexeMatrix* b);
ComplexeMatrix* mat_mul(ComplexeMatrix* a, ComplexeMatrix* b);
ComplexeMatrix* mat_scalar_div(ComplexeMatrix* m, double scalar);
ComplexeMatrix* matrix_exponential_fixed(ComplexeMatrix* A);
void time_evolution_diagonalization(
    const ComplexeMatrix* H_c, // Matrice Hamiltonien complexe (dim x dim)
    const cplx* psi0_c,        // Vecteur initial (dim)
    int dim,                   // Dimension
    const double* T_array,     // Tableau de temps (nbr_pts)
    int nbr_pts,               // Nombre de points temporels
    cplx* psi_out              // Sortie : [nbr_pts][dim] (row-major)
);

#ifdef __cplusplus
    }
#endif

#endif // UTILS_BACKUP_H