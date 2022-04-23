#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DEFAULTMAXITER 200
typedef enum
{
    false,
    true
} bool;

/*
allocates and initializes dynamic memory for matrix- array with size rows of arrays with size cols
*/
static double **create_matrix(int cols, int rows)
{
    double **mat = calloc(rows, sizeof(*mat)); //allocates the rows of the matrix
    if (mat==NULL) { //the calloc request failed
        printf("An Error Has Occurred");
        exit(1); //terminate and return 1
    }
    int row;
    for (row = 0; row < rows; row++) //for every row of the matrix
    {
        mat[row] = calloc(cols, sizeof(*mat[row])); //allocates column in the matrix
        if (mat[row]==NULL) { //the calloc request failed
        printf("An Error Has Occurred");
        exit(1); //terminate and return 1
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
    for (row = 0; row < rows; row++) //for every row in the matrix
    {
        free(matrix[row]); //free the column in the matrix
    }
    free(matrix); //free the rows of the matrix
}

/*
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the number of lines in in_file
*/
static int count_lines(FILE *in_file)
{
    int counter = 1; //counter of lines in the file sets to 1 (first line)
    char c; //the current char from the file
    while ((c = getc(in_file)) != EOF) //get next char until end of file
    {
        if (c == '\n') //the char represents new line
        {
            counter++; 
        }
    }
    rewind(in_file); //sets the file position to the beginning of the file
    return counter;
}

/*
in_file- input file for k-means algorithem: file that contains datapoints separated by commas
returns the dimension of the vectors in in_file
*/
static int find_dim(FILE *in_file)
{
    int counter = 1; //counter of lines in the file sets to 1 (first line)
    char c; //the current char from the file
    while ((c = getc(in_file)) != '\n') //get next char until end of first line
    {
        if (c == ',') //the char represnt new number in the vector
        {
            counter++; 
        }
    }
    rewind(in_file);  //sets the file position to the beginning of the file
    return counter;
}

/*
input_file_path- the input file for the K-means algorithem
vectors- pointer to the vectors variable that represents the number of vector in the input
dim- pointer to the dim variable that represents the dim of every vector in the input
opens the input file and return Two-dimensional array of the input
sets vectors and dim according to the input
*/
static double **read_input_file(char const *input_file_path, int *vectors, int *dim)
{
    FILE *in_file = fopen(input_file_path, "r"); //opens the input file
    *vectors = count_lines(in_file); //the number of lines in in_file
    *dim = find_dim(in_file); //the dimension of the vectors in in_file
    double **input_matrix = create_matrix(*vectors, *dim); //create the 
    int vector;
    for (vector = 0; vector < *vectors; vector++) //for every input vector
    {
        int num;
        for (num = 0; num < *dim; num++) //for every number in the input vector
        {
             //reads number from the input file and saves it in input_matrix
            fscanf(in_file, "%lf,", &input_matrix[vector][num]);
        }
        //reads the last number in the vector from the file
        fscanf(in_file, "%lf\n", &input_matrix[vector][num]); 
    }
    fclose(in_file); // closes the file
    return input_matrix;
}


static double **initialize_centroids(int k, int dim)
{
    FILE *in_file = fopen("first_centroids.txt", "r");
    double **centroids = create_matrix(k, dim);
    int i;
    for (i = 0; i < k; i++)
    {
        int j;
        for (j = 0; j < dim; j++)
        {
            fscanf(in_file, "%lf,", &centroids[i][j]);
        }
        fscanf(in_file, "%lf\n", &centroids[i][j]);
    }
    fclose(in_file);
    return centroids;
}
static double euclidean_norm(double *vector, int dim)
{
    double square_sum = 0;
    int i;
    for (i = 0; i < dim; i++)
    {
        square_sum += vector[i] * vector[i];
    }
    return sqrt(square_sum);
}
static double *add_vectors(double *new_vector, double *a, double *b, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
    {
        new_vector[i] = a[i] + b[i];
    }
    return new_vector;
}
static double *sub_vectors(double *new_vector, double *a, double *b, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
    {
        new_vector[i] = a[i] - b[i];
    }
    return new_vector;
}
static double dist(double *a, double *b, int dim)
{
    double *minus = calloc(dim, sizeof(*minus));
    double result;
    sub_vectors(minus, a, b, dim);
    result = euclidean_norm(minus, dim);
    free(minus);
    return result;
}
static int find_best_cluster(double **centroids, double *vector, int k, int dim)
{
    double min_dist = dist(centroids[0], vector, dim);
    int min_cluster = 0;
    int i;
    for (i = 1; i < k; i++)
    {
        double curr_dist = dist(vector, centroids[i], dim);
        if (curr_dist < min_dist)
        {
            min_cluster = i;
            min_dist = curr_dist;
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

void kmeans(int k, int max_iter, double epsilon, const char *input_file_path, const char *output_file_path)
{
    int dim;
    int vectors;
    double **input_vectors = read_input_file(input_file_path, &vectors, &dim);
    double **centroids = initialize_centroids(k, dim);
    double **clusters_sum = create_matrix(k, dim);
    double *clusters_lens = calloc(k, sizeof(double));
    int i;
    for (i = 0; i < max_iter; i++)
    {
        bool convergence = true;
        int vector_idx;
        int centroid_idx;
        for (vector_idx = 0; vector_idx < vectors; vector_idx++)
        {
            int best_cluster = find_best_cluster(centroids, input_vectors[vector_idx], k, dim);
            add_vectors(clusters_sum[best_cluster], clusters_sum[best_cluster], input_vectors[vector_idx], dim);
            (clusters_lens[best_cluster])++;
        }
        for (centroid_idx = 0; centroid_idx < k; centroid_idx++)
        {
            double *new_centroid = divide(clusters_sum[centroid_idx], clusters_lens[centroid_idx], dim);
            if (fabs(euclidean_norm(centroids[centroid_idx], dim) - euclidean_norm(new_centroid, dim)) > epsilon)
            {
                convergence = false;
            }
            free(centroids[centroid_idx]);
            centroids[centroid_idx] = new_centroid;
        }
        memset(clusters_lens, 0, k);
        matrix_reset(clusters_sum, k, dim);
    }
    write_output(output_file_path, centroids, k, dim);
    free_matrix(input_vectors, vectors);
    free_matrix(centroids, k);
    free_matrix(clusters_sum, k);
    free(clusters_lens);
}

int main() {
   return 0;
}