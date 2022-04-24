#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAX_ITER_JACOBI 100
#define MAX_ITER_KMEANS 300
#define EPSILON 1e-5
#define FIRST_CENTROIDS "first_centroids.txt"
#define RESULT_FILE "result.txt"

typedef enum
{
    false,
    true
} bool;

/*
handle invalid input:
*/
static void invalid_input()
{
    printf("Invalid Input!");
    exit(1); /* terminate */
}

/*
handle other errors (not invalid input):
*/
static void error()
{
    printf("An Error Has Occurred");
    exit(1); /* terminate */
}



/*
 *
 *basic functions- math, matrix:
 *
*/


/*
sign function, by the project definition
*/
double my_sign(double x) {
    if (x==0) {
        return 1;
    }
    return x/(fabs(x));
}

/*
allocates and initializes dynamic memory for matrix- array with size rows of arrays with size cols
*/
static double **create_matrix(int rows, int cols)
{
    int row;
    double **mat = calloc(rows, sizeof(*mat)); /* allocates the rows of the matrix */
    if (mat == NULL)
    { /* the calloc request failed */
        error();
    }
    for (row = 0; row < rows; row++) /* for every row of the matrix */
    {
        mat[row] = calloc(cols, sizeof(*mat[row])); /* allocates column in the matrix */
        if (mat[row] == NULL)
        { /* the calloc request failed */
            error();
        }
    }
    return mat;
}

/*
mat- array with size rows of arrays
free the allocated memory of mat
*/
static void free_matrix(double **matrix, double rows)
{
    int row;
    for (row = 0; row < rows; row++) /* for every row in the matrix */
    {
        free(matrix[row]); /* free the column in the matrix */
    }
    free(matrix); /* free the rows of the matrix */
}

/*
calculates and returns euclidean norm of vector with dimention dim
*/
static double euclidean_norm(double *vector, int dim)
{
    double square_sum = 0; /*the sum of the square of every num in the vector*/
    int i; /*index for every number in the vector*/
    for (i = 0; i < dim; i++)
    {
        square_sum += vector[i] * vector[i]; /*addes the square of vector[i] to the sum*/
    }
    return sqrt(square_sum); /*calculates square root of the sum*/
}

/*
sum vectors a, b with dimention dim into new_vector
*/
static void add_vectors(double *new_vector, double *a, double *b, int dim)
{
    int i;
    for (i = 0; i < dim; i++) 
    {
        new_vector[i] = a[i] + b[i]; /*sum every number in the vectors*/
    }
}

/*
subtract vectors a, b with dimention dim into new_vector
*/
static void sub_vectors(double **new_vector, double *a, double *b, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
    {
        (*new_vector)[i] = a[i] - b[i]; /*substract every number in the vectors*/
    }
}

/*
calculate  the distance between a, b with dimention dim
*/
static double dist(double *a, double *b, int dim)
{
    double *minus = calloc(dim, sizeof(double)); /*a-b vector*/
    double result; 
    sub_vectors(&minus, a, b, dim);
    result = euclidean_norm(minus, dim);
    free(minus);
    return result;
}

/*
dim- the dimension of vec
calculates and return vec/d
*/
static double *divide(double *a, double d, int dim)
{
    double *result = calloc(dim, sizeof(*result));
    int i;
    for (i = 0; i < dim; i++)
    {
        result[i] = a[i] / d;
    }
    return result;
}

/*
mat1, mat2, res are mateixes with n rows and n columns
calculate multiplication of mat1, mat2 and save it in res
*/
static void multiply(double **mat1, double **mat2, double **res, int n)
{
   int i, j, k;
   for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
         res[i][j] = 0;
         for (k = 0; k < n; k++)
            res[i][j] += mat1[i][k] * mat2[k][j];
        }
    }
}

/*
reset- initialize matrix to 0
*/
static void matrix_reset(double **matrix, int rows, int cols)
{
    int row;
    for (row = 0; row < rows; row++)
    {
        memset(matrix[row], 0, cols * sizeof(double));
    }
}



/*
 *
 *input, output and files functions:
 *
 */


/*
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the number of lines in in_file
*/
static int count_lines(FILE *in_file)
{
    int counter = 0; /*counter of lines in the file sets to 0*/
    char c; /*the current char from the file*/
    while ((c = getc(in_file)) != EOF) /*get next char until end of file*/
    {
        if (c == '\n') /*the char represents new line*/
        {
            counter++;
        }
    }
    rewind(in_file); /* sets the file position to the beginning of the file */
    return counter;
}

/*
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the dimension of the vectors in in_file
*/
static int find_dim(FILE *in_file)
{
    int counter = 1; /*counter of lines in the file sets to 1 (first line)*/
    char c; /*the current char from the file*/
    while ((c = getc(in_file)) != '\n') /*get next char until end of first line*/
    {
        if (c == ',') /*the char represnt new number in the vector*/
        {
            counter++;
        }
    }
    rewind(in_file);  /*sets the file position to the beginning of the file*/
    return counter;
}

/*
in_file- path to input file that contain vectors
lines- the amount of lines in the file
dim- the dim of every vector in the file
opens the input file and return Two-dimensional array of the input vectors
*/
static double **read_vectors_file(FILE *in_file, int lines, int dim){
    double **input_matrix = create_matrix(lines, dim); /*create the input matrix*/
    int scan_res; /*the returned value of fscanf function*/
    int vector;
    for (vector = 0; vector < lines; vector++) /*for every input vector*/
    {
        int num;
        for (num = 0; num < dim-1; num++) /*for every number in the input vector*/
        {
             /*reads number from the input file and saves it in input_matrix*/
            scan_res = fscanf(in_file, "%lf,", &input_matrix[vector][num]);
            if (scan_res == EOF)
            {
                invalid_input();
            }
        }
        /* reads the last number in the vector from the file */
        scan_res = fscanf(in_file, "%lf\n", &input_matrix[vector][num]);
        if (scan_res == EOF)
        {
            invalid_input();
        }
    }
    fclose(in_file); /* closes the file */
    return input_matrix;
}

/*
write output to output_file_path (for the python program)
*/
static void write_output(double **result_marix, int rows, int cols)
{
    FILE *out_file = fopen(RESULT_FILE, "w"); /*open new file*/
    int row;
    for (row = 0; row < rows; row++)
    {
        int col;
        for (col = 0; col < cols-1; col++)
        {
            fprintf(out_file, "%.4f,", result_marix[row][col]); /*number in matrix + ,*/
        }
        fprintf(out_file, "%.4f\n", result_marix[row][col]); /*last number in line + end of line*/
    }
    fclose(out_file);
}

/*
print output
*/




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
static int find_best_cluster(double **centroids, double *vector, int k, int dim)
{
    /*minimal distance between vector from centroid in centroids:*/
    double min_dist = dist(centroids[0], vector, dim);
    /*index of the centroid that has min_dist distance from vector*/ 
    int min_cluster = 0;
    int centroid_idx;
    for (centroid_idx = 1; centroid_idx < k; centroid_idx++)
    {
        /*distance between vector to current centroid (centroids[centroid_idx])*/
        double curr_dist = dist(vector, centroids[centroid_idx], dim); 
        if (curr_dist < min_dist) 
        {
            min_dist = curr_dist; /*update minimal distance*/
            min_cluster = centroid_idx; /*update matching centroid index*/
        }
    }
    return min_cluster;
}

/*
Kmeans: read first input from input_file_path, sets c_vectors&dim
*/
double **read_input(int *c_vectors, int *dim, const char *input_file_path) {
    FILE *in_file = fopen(input_file_path, "r"); /*opens the input file*/
    if (in_file==NULL){ 
        invalid_input();
    }
    *c_vectors = count_lines(in_file); 
    *dim = find_dim(in_file);
    return read_vectors_file(in_file, *c_vectors, *dim);
}

void kmeans(int k, double **input_vectors)
{
    int c_vectors=0; /*the amount of input vectors*/
    int dim=0; /*the dimension of the vectors in in_file*/
    double **centroids; 
    double **clusters_sum; /*clusters_sum[i] is the sum vector of the vectors in cluster i*/
    double *clusters_lens; /*clusters_sum[i] is the amount of vectors in cluster i*/
    /*initializes centroids by reading the file that the python program created:*/
    FILE *centroids_file;
    int count_iter;
    centroids_file = fopen(FIRST_CENTROIDS, "r"); /*opens the input file*/
    if (centroids_file==NULL){ 
        invalid_input();
    }
    centroids = read_vectors_file(centroids_file, k, dim);
    clusters_sum = create_matrix(k, dim);
    clusters_lens = calloc(k, sizeof(double));
    /*until convergence OR iteration number = max iter*/
    for (count_iter = 0; count_iter < MAX_ITER_KMEANS; count_iter++)
    {
        bool convergence = true; 
        int vector_idx;
        int centroid_idx;
        for (vector_idx = 0; vector_idx < c_vectors; vector_idx++)
        {
            /*assign input vector x_i to the cluster S_j = increases cluster len and adds to sum:*/
            int best_cluster = find_best_cluster(centroids, input_vectors[vector_idx], k, dim);
            add_vectors(clusters_sum[best_cluster], clusters_sum[best_cluster], input_vectors[vector_idx], dim);
            (clusters_lens[best_cluster])++;
        }
        /*update the centroids:*/
        for (centroid_idx = 0; centroid_idx < k; centroid_idx++)
        {
            double *new_centroid = divide(clusters_sum[centroid_idx], clusters_lens[centroid_idx], dim);
            /*convergence <-> Euclidean norm for each one of the centroids doesn’t change*/
            if (fabs(euclidean_norm(centroids[centroid_idx], dim) - euclidean_norm(new_centroid, dim)) > 0) 
            {
                convergence = false;
            }
            free(centroids[centroid_idx]);
            centroids[centroid_idx] = new_centroid;
        }
        if (convergence == true)
        {
            break;
        }
        /*reset clusters for next iteration:*/
        memset(clusters_lens, 0, k);
        matrix_reset(clusters_sum, k, dim);
    }
    write_output(centroids, k, dim);
    free_matrix(input_vectors, c_vectors);
    free_matrix(centroids, k);
    free_matrix(clusters_sum, k);
    free(clusters_lens);
}



/*
 *
 *Spectral clustering functions:
 *
 */ 


/*
1: compute the weighted adjacency matrix Wadj with the graph param (n columns and n rows)
*/
static double **create_Wadj(double **matrix, int n, int dim)
{
    double val; /*value in the matrix*/
    double **Wadj = create_matrix(n, n);
    int i;
    for (i = 0; i < n; i++)
    {
        int j;
        for (j = 0; j < i; j++)
        {
            val = exp(dist(matrix[i], matrix[j], dim) / (-2));
            Wadj[i][j] = val;
            Wadj[j][i] = val;
        }
    }
    return Wadj;
}

/*
2: Compute the The Diagonal Degree Matrix Lnorm with the weights param (n columns and n rows)
*/
static double **create_Dsqrt(double **weights, int n)
{
    double **Dsqrt = create_matrix(n, n);
    /* calculate Dsqrt: */
    int row;
    for (row = 0; row < n; row++)
    {
        int col;
        for (col = 0; col < n; col++)
        {
            Dsqrt[row][row] += weights[row][col];
        }
        Dsqrt[row][row] = 1/sqrt(Dsqrt[row][row]);
    }
    return Dsqrt;
}

/*
3: Compute the normalized graph Laplacian Lnorm with the weights param (n columns and n rows)
*/
static double **create_Lnorm(double **weights, double **Dsqrt, int n)
{
    double **Lnorm = create_matrix(n, n); /* this will be the final result */
    int row;
    /* clculate Lnorm: */
    for (row = 0; row < n; row++)
    {
        int col;
        for (col = 0; col < n; col++)
        {
            Lnorm[row][col] = -(weights[row][col] * Dsqrt[row][row] * Dsqrt[col][col]);
        }
        Lnorm[row][row] = 1 + Lnorm[row][row];
    }
    return Lnorm;
}

/*
5: Form the matrix T ∈ Rn×k
from U by renormalizing each of U’s rows to have unit length
*/
static double **create_T(double **U, int n)
{
    int row;
    double **T = create_matrix(n, n);
    for (row = 0; row < n; row++)
    {
        T[row] = divide(U[row], euclidean_norm(U[row], n), n);
    }
    return T;
}

/*
diagonal_mat- Lnorm diagonal matrix (with eigenvalues on diagonal)
n- size of diagonal_mat
determine the number of clusters k
*/
int Eigengap_Heuristic(double **diagonal_mat, int n) {
    double max_dist=0; /*maximal dist between eigenvalue-i, eigenvalue-i+1*/
    int max_i=0; /*matching index for maximal dist*/
    int i;
    double curr_dist;
    for (i=0; i<floor(n/2); i++) {
        curr_dist = fabs(diagonal_mat[i][i]-diagonal_mat[i+1][i+1]);
        if (curr_dist>max_dist) {
            max_dist = curr_dist;
            max_i = i;
        }
    }
    return max_i;
}



/*
 *
 *Jacobi algorithm functions:
 *
 */ 


/*
n- the amount of rows in the matrix, the amount of cols in the matrix
calculate theta in Jacobi eigenvalue algorithm
*/
static double clac_theta(double **A, int n, int *max_i, int *max_j) {
    /*find pivot:*/
    double max_val=0; /*the max absolute value, and the matching indexes*/
    int curr_i, curr_j; /*current indexes in the matrix*/
    double curr_val; /*current value in the matrix*/
    for (curr_i=0; curr_i<n; curr_i++) {
        for (curr_j=0; curr_j<n; curr_j++) {
            if (curr_i!=curr_j) { /*off-diagonal elements only*/
                curr_val = fabs(A[curr_i][curr_j]);
                if (curr_val > max_val) {
                    max_val = curr_val;
                    *max_i = curr_i;
                    *max_j = curr_j;
                }
            }
        }
    }
    return (A[*max_j][*max_j]-A[*max_i][*max_i])/(2*A[*max_i][*max_j]);
}

/*
calculates and returns off(A)^2-off(A')^2 for Jacobi eigenvalue algorithm
*/
static double off_diff(double **A, double **new_A, int n) {
    double off_A = 0;
    double off_new = 0;
    int row; int col;
    for (row=0; row<n; row++) {
        for (col=0; col<n; col++) {
            if (row!=col) {
                off_A+= A[row][col]*A[row][col];
                off_new+= new_A[row][col]*new_A[row][col];
            }
        }
    }
    return off_A-off_new;
}

/*
n- the amount of rows in the matrix, the amount of cols in the matrix
The Jacobi eigenvalue algorithm is an iterative method for the calculation of the eigenvalues and
eigenvectors of a real symmetric matrix (a process known as diagonalization).
*/
static double **jacobi(double **Lnorm, int n, double ***final_A) {
    double **temp_mult = create_matrix(n, n); /*variable for temporary results*/
    double **V; /*the result matrix*/
    double **A; /*the current matrix*/
    double **new_A = Lnorm; /*A'- the next matrix*/
    double **P = create_matrix(n, n); /*the Jacobi rotation matrix*/
    int max_i, max_j; /*the i,j indexes from the calculation of theta*/
    double theta, t, c, s;
    int count_iter =0;
    int mat_idx;
    while ((count_iter==0) || ((count_iter<MAX_ITER_JACOBI) && (off_diff(A, new_A, n)>EPSILON))) {
        if (count_iter!=0) {
            free_matrix(A, n);
        }
        A = new_A; 
        matrix_reset(P, n, n); 
        theta = clac_theta(A, n, &max_i, &max_j);
        t = (my_sign(theta))/(fabs(theta)+sqrt(theta*theta+1));
        c = 1/(sqrt(t*t+1));
        s = t*c;
        for (mat_idx=0; mat_idx<n; mat_idx++){ /*set diagonal values in P*/
            P[mat_idx][mat_idx] =1;
        }
        /*set values in P:*/
        P[max_i][max_i] =c; P[max_j][max_j] =c; P[max_i][max_j] =s; P[max_j][max_i] =-s;
        /*update V:*/
        if (count_iter==0) {
            V=P;
        }
        else{
            multiply(V, P, temp_mult, n);
            V=temp_mult; 
        }
        /*calculate A':*/
        new_A = create_matrix(n, n);
        for (mat_idx=0; mat_idx<n; mat_idx++) {
            new_A[mat_idx][max_i] = c*A[mat_idx][max_i]-s*A[mat_idx][max_j];
            new_A[mat_idx][max_j] = c*A[mat_idx][max_j]-s*A[mat_idx][max_i];
        }
        new_A[max_i][max_i]= c*c*A[max_i][max_i] + s*s*A[max_j][max_j]-2*s*c*A[max_i][max_j];
        new_A[max_j][max_j]= s*s*A[max_i][max_i] + c*c*A[max_j][max_j]-2*s*c*A[max_i][max_j];
        new_A[max_i][max_j] = 0; new_A[max_j][max_i] = 0;
        count_iter++;
    }
    *final_A = new_A;
    free_matrix(P, n);
    return V;
}





/*
preform the algorithm that required in goal
*/
void algorithm(const char *goal, const char *file_path, int k) {
    int rows, cols;
    double **X= read_input(&rows, &cols, file_path);
    double **W= create_Wadj(X, rows, cols);
    double **Dsqrt, **L, **final_A, **U, **T;
    free_matrix(X, rows);
    if (strcmp(goal, "wam")==0){
        write_output(W, rows, rows);
        free_matrix(W, rows);
        return;
    }
    Dsqrt = create_Dsqrt(W, rows);
    if (strcmp(goal, "ddg")==0){
        write_output(Dsqrt, rows, rows);
        free_matrix(Dsqrt, rows); free_matrix(W, rows);
        return;
    }
    L = create_Lnorm(W, Dsqrt, rows);
    free_matrix(Dsqrt,rows);
    if (strcmp(goal, "lnorm")==0){
        write_output(L, rows, rows);
        free_matrix(L,rows);
        return;
    }
    final_A = create_matrix(rows, rows);
    U = jacobi(L, rows, &final_A);
    if (strcmp(goal, "jacobi")==0){
        write_output(U, rows, rows);
        free_matrix(U, rows);
        free_matrix(final_A, rows); 
        return;
    }
    T = create_T(U, rows);
    free_matrix(U, rows);
    if (strcmp(goal, "T")==0) {
        write_output(T, rows, rows);
        free_matrix(T,rows); free_matrix(final_A,rows);
        return;
    }
    free_matrix(T, rows);
    if (strcmp(goal, "spk")==0) {
        if (k==0) {
            k = Eigengap_Heuristic(final_A, rows);
        }
        kmeans(k, T);
    }
}


int main(int argc, const char **argv)
{
    if (argc == 3)
    {
        const char *goal = argv[1]; 
        /*check file name:*/
        const char *mime = strchr(argv[2], '.');
        if ((mime != NULL && (strcmp(mime, ".txt") == 0 || strcmp(mime, ".csv") == 0))
        && (strcmp(goal, "wadj") || strcmp(goal, "ddg") || strcmp(goal, "lnorm") || strcmp(goal, "jacobi"))) 
        {
            algorithm(goal, argv[2], 0);
            return 0;
        }
    }
    invalid_input();
    return 1;
}
