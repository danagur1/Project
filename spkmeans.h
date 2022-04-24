/*
handle invalid input:
*/
void invalid_input();

/*
handle other errors (not invalid input):
*/
void error();



/*
 *
 *basic functions- math, matrix:
 *
*/


/*
sign function, by the project definition
*/
double my_sign(double x);
/*
allocates and initializes dynamic memory for matrix- array with size rows of arrays with size cols
*/
double **create_matrix(int rows, int cols);
/*
mat- array with size rows of arrays
free the allocated memory of mat
*/
void free_matrix(double **matrix, double rows);
/*
calculates and returns euclidean norm of vector with dimention dim
*/
double euclidean_norm(double *vector, int dim);
/*
sum vectors a, b with dimention dim into new_vector
*/
void add_vectors(double *new_vector, double *a, double *b, int dim);
/*
subtract vectors a, b with dimention dim into new_vector
*/
void sub_vectors(double **new_vector, double *a, double *b, int dim);
/*
calculate  the distance between a, b with dimention dim
*/
double dist(double *a, double *b, int dim);
/*
dim- the dimension of vec
calculates and return vec/d
*/
double *divide(double *a, double d, int dim);
/*
mat1, mat2, res are mateixes with n rows and n columns
calculate multiplication of mat1, mat2 and save it in res
*/
void multiply(double **mat1, double **mat2, double **res, int n);
/*
reset- initialize matrix to 0
*/
void matrix_reset(double **matrix, int rows, int cols);



/*
 *
 *input, output and files functions:
 *
 */


/*
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the number of lines in in_file
*/
int count_lines(FILE *in_file);
/*
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the dimension of the vectors in in_file
*/
int find_dim(FILE *in_file);
/*
in_file- path to input file that contain vectors
lines- the amount of lines in the file
dim- the dim of every vector in the file
opens the input file and return Two-dimensional array of the input vectors
*/
double **read_vectors_file(FILE *in_file, int lines, int dim);
/*
write output to output_file_path (for the python program)
*/
void write_output(double **result_marix, int rows, int cols);
/*
nXm- size of matrix
print output
*/
void print_output_matrix(double **matrix, int n, int m);



/*
 *
 *K-means algorithm functions:
 *
 */ 


/*
k- the amount of centroids
dim- the dimension of the vector
in K-means algorithm- finds and returns the index of the closet cluster (in centroids) to vector
*/
int find_best_cluster(double **centroids, double *vector, int k, int dim);

/*
Kmeans: read first input from input_file_path, sets c_vectors&dim
*/
double **read_input(int *c_vectors, int *dim, const char *input_file_path);

void kmeans(int k, double **input_vectors);



/*
 *
 *Spectral clustering functions:
 *
 */ 


/*
1: compute the weighted adjacency matrix Wadj with the graph param (n columns and n rows)
*/
double **create_Wadj(double **matrix, int n, int dim);

/*
2: Compute the The Diagonal Degree Matrix Lnorm with the weights param (n columns and n rows)
*/
double **create_Dsqrt(double **weights, int n);

/*
3: Compute the normalized graph Laplacian Lnorm with the weights param (n columns and n rows)
*/
double **create_Lnorm(double **weights, double **Dsqrt, int n);

/*
5: Form the matrix T ∈ Rn×k
from U by renormalizing each of U’s rows to have unit length
*/
double **create_T(double **U, int n);

/*
diagonal_mat- Lnorm diagonal matrix (with eigenvalues on diagonal)
n- size of diagonal_mat
determine the number of clusters k
*/
int Eigengap_Heuristic(double **diagonal_mat, int n);



/*
 *
 *Jacobi algorithm functions:
 *
 */ 


/*
n- the amount of rows in the matrix, the amount of cols in the matrix
calculate theta in Jacobi eigenvalue algorithm
*/
double clac_theta(double **A, int n, int *max_i, int *max_j);

/*
calculates and returns off(A)^2-off(A')^2 for Jacobi eigenvalue algorithm
*/
double off_diff(double **A, double **new_A, int n);

/*
n- the amount of rows in the matrix, the amount of cols in the matrix
The Jacobi eigenvalue algorithm is an iterative method for the calculation of the eigenvalues and
eigenvectors of a real symmetric matrix (a process known as diagonalization).
*/
double **jacobi(double **Lnorm, int n, double ***final_A);





/*
preform the algorithm that required in goal
*/
void algorithm(const char *goal, const char *file_path, int k, void output_format(double **, int, int));