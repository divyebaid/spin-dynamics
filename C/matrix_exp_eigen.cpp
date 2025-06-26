#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "utils_backup.h"
#include <omp.h>  // Ajoute ça !


using namespace Eigen;

static inline MatrixXcd C_to_Eigen(const ComplexeMatrix* cmat) {
    int n = cmat->dim;
    MatrixXcd mat(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat(i, j) = cmat->compMatrix[i][j];
    return mat;
}

static inline void Eigen_to_C(const MatrixXcd& mat, ComplexeMatrix* cmat) {
    int n = cmat->dim;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cmat->compMatrix[i][j] = mat(i, j);
}

extern "C" {

void time_evolution_diagonalization(
    const ComplexeMatrix* H_c,
    const cplx* psi0_c,
    int dim,
    const double* T_array,
    int nbr_pts,
    cplx* psi_out  // [nbr_pts][dim], row-major
) {
    using namespace Eigen;
    using std::complex;

    // Convertit H (C->Eigen)
    MatrixXcd H(dim, dim);
    for (int i=0; i<dim; ++i)
        for (int j=0; j<dim; ++j)
            H(i,j) = H_c->compMatrix[i][j];

    // Diagonalise H (une seule fois)
    ComplexEigenSolver<MatrixXcd> ces(H, true);
    MatrixXcd V = ces.eigenvectors();
    MatrixXcd Vinv = V.inverse();
    VectorXcd evals = ces.eigenvalues();

    // psi0 in Eigen
    VectorXcd psi0(dim);
    for (int i=0; i<dim; ++i) psi0(i) = psi0_c[i];

    // Projette psi0 dans la base propre
    VectorXcd psi_diag = Vinv * psi0;

    constexpr double HBAR = 1.054571817e-34;

    // Parallélisation OpenMP sur t_idx
    #pragma omp parallel for schedule(static)
    for (int t_idx = 0; t_idx < nbr_pts; ++t_idx) {
        double t = T_array[t_idx];
        VectorXcd phase = (-complex<double>(0,1) * evals * t / HBAR).array().exp();
        VectorXcd psi_diag_t = psi_diag.array() * phase.array();
        VectorXcd psi_t = V * psi_diag_t;
        for (int i=0; i<dim; ++i)
            psi_out[t_idx*dim + i] = psi_t(i);
    }
}


ComplexeMatrix* allocate_complex_matrix(int dim) {
    ComplexeMatrix* m = (ComplexeMatrix*)malloc(sizeof(ComplexeMatrix));
    m->dim = dim;
    m->compMatrix = (cplx**)malloc(dim * sizeof(cplx*));
    for (int i = 0; i < dim; ++i)
        m->compMatrix[i] = (cplx*)calloc(dim, sizeof(cplx));
    return m;
}

void free_complex_matrix(ComplexeMatrix* matrix) {
    if (!matrix) return;
    for (int i = 0; i < matrix->dim; ++i)
        free(matrix->compMatrix[i]);
    free(matrix->compMatrix);
    free(matrix);
}

ComplexeMatrix* mat_identity(int dim) {
    ComplexeMatrix* out = allocate_complex_matrix(dim);
    MatrixXcd eye = MatrixXcd::Identity(dim, dim);
    Eigen_to_C(eye, out);
    return out;
}

ComplexeMatrix* mat_add(ComplexeMatrix* a, ComplexeMatrix* b) {
    if (!a || !b || a->dim != b->dim) return NULL;
    int dim = a->dim;
    ComplexeMatrix* out = allocate_complex_matrix(dim);
    MatrixXcd A = C_to_Eigen(a);
    MatrixXcd B = C_to_Eigen(b);
    MatrixXcd C = A + B;
    Eigen_to_C(C, out);
    return out;
}

ComplexeMatrix* mat_mul(ComplexeMatrix* a, ComplexeMatrix* b) {
    if (!a || !b || a->dim != b->dim) return NULL;
    int dim = a->dim;
    ComplexeMatrix* out = allocate_complex_matrix(dim);
    MatrixXcd A = C_to_Eigen(a);
    MatrixXcd B = C_to_Eigen(b);
    MatrixXcd C = A * B;
    Eigen_to_C(C, out);
    return out;
}

ComplexeMatrix* mat_scalar_div(ComplexeMatrix* m, double scalar) {
    MatrixXcd M = C_to_Eigen(m);
    M /= scalar;
    Eigen_to_C(M, m);
    return m;
}

// Exponentielle de matrice (exactement comme avant)
ComplexeMatrix* matrix_exponential_fixed(ComplexeMatrix* A) {
    int dim = A->dim;
    ComplexeMatrix* result = allocate_complex_matrix(dim);
    MatrixXcd mat = C_to_Eigen(A);
    MatrixXcd expmat = mat.exp();
    Eigen_to_C(expmat, result);
    return result;
}

} // extern "C"