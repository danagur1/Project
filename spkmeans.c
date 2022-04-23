#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAX_ITER 200
#define EPSILON 1e-6
#define FIRST_CENTROIDS "first_centroids.txt"

typedef enum
{
    false,
    true
} bool;

/*
handle invalid input
*/
static void invalid_input()
{
    printf("Invalid Input");
    exit(1); /* terminate */
}

/*
handle other errors (not invalid input)
*/
static void error()
{
    printf("An Error Has Occurred");
    exit(1); /* terminate */
}
/*
allocates and initializes dynamic memory for matrix- array with size rows of arrays with size cols
*/
static double **create_matrix(int cols, int rows)
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
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the number of lines in in_file
*/
static int count_lines(FILE *in_file)
{
    int counter = 1; /*counter of lines in the file sets to 1 (first line)*/
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
        for (num = 0; num < dim; num++) /*for every number in the input vector*/
        {
             /*reads number from the input file and saves it in input_matrix*/
            scan_res = fscanf(in_file, "%lf,", &input_matrix[vector][num]);
            if (scan_res == 0)
            {
                invalid_input();
            }
        }
        /* reads the last number in the vector from the file */
        fscanf(in_file, "%lf\n", &input_matrix[vector][num]);
        if (scan_res == 0)
        {
            invalid_input();
        }
    }
    fclose(in_file); /* closes the file */
    return input_matrix;
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
static void sub_vectors(double *new_vector, double *a, double *b, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
    {
        new_vector[i] = a[i] - b[i]; /*substract every number in the vectors*/
    }
}

/*
calculate  the distance between a, b with dimention dim
*/
static double dist(double *a, double *b, int dim)
{
    double *minus = calloc(dim, sizeof(*minus)); /*a-b vector*/
    double result; 
    sub_vectors(minus, a, b, dim);
    result = euclidean_norm(minus, dim);
    free(minus);
    return result;
}

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

static void write_output(const char *output_file_path, double **centroids, int k, int dim)
{
    FILE *out_file = fopen(output_file_path, "w");
    int i;
    for (i = 0; i < k; i++)
    {
        int j;
        for (j = 0; j < dim; j++)
        {
            fprintf(out_file, "%.4f,", centroids[i][j]);
        }
        fprintf(out_file, "%.4f\n", centroids[i][j]);
    }
    fclose(out_file);
}

static void matrix_reset(double **matrix, int k, int dim)
{
    int i;
    for (i = 0; i < k; i++)
    {
        memset(matrix[i], 0, dim * sizeof(*matrix[i]));
    }
}

/*
1: compute the weighted adjacency matrix Wadj with the graph param (n columns and n rows)
*/
static double **create_Wadj(double **matrix, int n)
{
    double **Wadj = create_matrix(n, n);
    int i;
    for (i = 0; i < n; i++)
    {
        int j;
        for (j = 0; j < i; j++)
        {
            Wadj[i][j] = exp(dist(matrix[i], matrix[j], n) / -2);
        }
    }
    return Wadj;
}

/*
2: Compute the The Diagonal Degree Matrix Lnorm with the weights param (n columns and n rows)
*/
static double *create_D(double **weights, int n)
{
    double *D_diagonal = calloc(n, sizeof(*D_diagonal));
    /* calculate D_diagonal: */
    int row;
    for (row = 0; row < n; row++)
    {
        int col;
        for (col = 0; col < n; col++)
        {
            D_diagonal[row] += weights[row][col];
        }
    }
    return D_diagonal;
}

/*
3: Compute the normalized graph Laplacian Lnorm with the weights param (n columns and n rows)
*/
static double **create_Lnorm(double **weights, double *D_diagonal, int n)
{
    double **Lnorm = create_matrix(n, n); /* this will be the final result */
    int row;
    /* clculate Lnorm: */
    for (row = 0; row < n; row++)
    {
        int col;
        for (col = 0; col < n; col++)
        {
            Lnorm[row][col] = weights[row][col] / sqrt(D_diagonal[row]) / sqrt(D_diagonal[col]);
        }
        Lnorm[row][row] = 1 - Lnorm[row][row];
    }
    return Lnorm;
}

/*
5: Form the matrix T ∈ Rn×k
from U by renormalizing each of U’s rows to have unit length
*/
static double **create_T(double **U, int rows, int cols)
{
    int row;
    double **T = create_matrix(rows, cols);
    for (row = 0; row < rows; row++)
    {
        T[row] = divide(U[row], euclidean_norm(U[row], cols), cols);
    }
    return T;
}

/*
n- the amount of rows in the matrix, the amount of cols in the matrix
calculate theta in Jacobi eigenvalue algorithm
*/
static double clac_theta(double **A, int n, int *max_i, int *max_j) {
    /*find pivot:*/
    int max_val=0; /*the max absolute value, and the matching indexes*/
    int curr_val, curr_i, curr_j; /*current value and indexes in the matrix*/
    for (curr_i=0; curr_i<n; curr_i++) {
        for (curr_j=0; curr_j<n; curr_j++) {
            curr_val = fabs(A[curr_i][curr_j]);
            if (curr_val > max_val) {
                max_val = curr_val;
                *max_i = curr_i;
                *max_j = curr_j;
            }
        }
    }
    return (A[*max_j][*max_j]-A[*max_i][*max_i])/(2*A[*max_i][*max_j]);
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
calculates and returns off(A)^2-off(A')^2 for Jacobi eigenvalue algorithm
*/
static double off_diff(double **A, double **new_A, int n) {
    double off_A = 0;
    double off_new = 0;
    int row; int col;
    for (row=0; row<n; row++) {
        for (col=0; col<n; col++) {
            if (row!=col) {
                off_A+= A[row][col];
                off_new+= new_A[row][col];
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
static double **jacobi(double **Lnorm, int n) {
    double **temp_mult = create_matrix(n, n); /*variable for temporary results*/
    double **V = create_matrix(n, n); /*the result matrix*/
    double **A = Lnorm; /*the current matrix*/
    double **new_A = A; /*A'- the next matrix*/
    double **P; /*the Jacobi rotation matrix*/
    int max_i, max_j; /*the i,j indexes from the calculation of theta*/
    double thetha, t, c, s;
    int count_iter =0;
    int mat_idx;
    while ((count_iter>1) && (count_iter<MAX_ITER) && (off_diff(A, new_A, n)>EPSILON)) {
        free(A);
        A = new_A; 
        P = create_matrix(n, n); 
        thetha = clac_theta(A, n, &max_i, &max_j);
        t = (_copysignf(1.0, thetha))/(fabs(thetha)+sqrt(thetha*thetha+1));
        c = 1/(sqrt(t*t+1));
        s = t*c;
        for (mat_idx=0; mat_idx<n; mat_idx++){ /*set diagonal values in P*/
            P[mat_idx][mat_idx] =1;
        }
        /*set values in P:*/
        P[max_i][max_i] =c; P[max_j][max_j] =c; P[max_i][max_j] =s; P[max_j][max_i] =s;
        multiply(V, P, temp_mult, n); V=P; /*update V*/
        /*calculate A':*/
        new_A = create_matrix(n, n);
        for (mat_idx=0; mat_idx<n; mat_idx++) {
            new_A[mat_idx][max_i] = c*A[mat_idx][max_i]-s*A[mat_idx][max_j];
            new_A[mat_idx][max_j] = c*A[mat_idx][max_j]-s*A[mat_idx][max_i];
        }
        new_A[max_i][max_i]= c*c*A[max_i][max_i] + s*s*A[max_j][max_j]-2*s*c*A[max_i][max_j];
        new_A[max_j][max_j]= s*s*A[max_i][max_i] + c*c*A[max_j][max_j]-2*s*c*A[max_i][max_j];
        new_A[max_i][max_j] = 0; new_A[max_j][max_i] = 0;
        free(P);
    }
    free(temp_mult);
    return V;
}

double **read_input(int *c_vectors, int *dim, const char *input_file_path) {
    FILE *in_file = fopen(input_file_path, "r"); /*opens the input file*/
    if (in_file==NULL){ 
        invalid_input();
    }
    *c_vectors = count_lines(in_file); 
    *dim = find_dim(in_file);
    return read_vectors_file(in_file, *c_vectors, *dim);
}

void kmeans(int k, const char *input_file_path, const char *output_file_path)
{
    int c_vectors; /*the amount of input vectors*/
    int dim; /*the dimension of the vectors in in_file*/
    double **input_vectors = read_input(&c_vectors, &dim, input_file_path);
    double **centroids; 
    double **clusters_sum; /*clusters_sum[i] is the sum vector of the vectors in cluster i*/
    double *clusters_lens; /*clusters_sum[i] is the amount of vectors in cluster i*/
    /*initializes centroids by reading the file that the python program created:*/
    FILE *centroids_file = fopen(FIRST_CENTROIDS, "r"); /*opens the input file*/
    if (centroids_file==NULL){ 
        invalid_input();
    }
    centroids = read_vectors_file(centroids_file, k, dim);
    clusters_sum = create_matrix(k, dim);
    clusters_lens = calloc(k, sizeof(double));
    int i;
    fclose(centroids_file);
    for (i = 0; i < MAX_ITER; i++)
    {
        bool convergence = true;
        int vector_idx;
        int centroid_idx;
        for (vector_idx = 0; vector_idx < c_vectors; vector_idx++)
        {
            int best_cluster = find_best_cluster(centroids, input_vectors[vector_idx], k, dim);
            add_vectors(clusters_sum[best_cluster], clusters_sum[best_cluster], input_vectors[vector_idx], dim);
            (clusters_lens[best_cluster])++;
        }
        for (centroid_idx = 0; centroid_idx < k; centroid_idx++)
        {
            double *new_centroid = divide(clusters_sum[centroid_idx], clusters_lens[centroid_idx], dim);
            if (fabs(euclidean_norm(centroids[centroid_idx], dim) - euclidean_norm(new_centroid, dim)) > EPSILON)
            {
                convergence = false;
            }
            free(centroids[centroid_idx]);
            centroids[centroid_idx] = new_centroid;
        }
        if (convergence)
        {
            break;
        }
        memset(clusters_lens, 0, k);
        matrix_reset(clusters_sum, k, dim);
    }
    write_output(output_file_path, centroids, k, dim);
    free_matrix(input_vectors, c_vectors);
    free_matrix(centroids, k);
    free_matrix(clusters_sum, k);
    free(clusters_lens);
}

int main(int argc, const char **argv)
{
    if (argc == 3)
    {
        int k = atoi(argv[1]);
        const char *mime = strchr(argv[2], '.');
        if (k > 0 && mime != NULL && (strcmp(mime, ".txt") == 0 || strcmp(mime, ".csv") == 0))
        {
            /* spkmeans algorithm... */
            kmeans(k, MAX_ITER, EPSILON, argv[2], FIRST_CENTROIDS);
            return 0;
        }
    }
    invalid_input();
    return 1;
}
