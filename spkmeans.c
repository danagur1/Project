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
    double **mat = calloc(rows, sizeof(*mat));
    int row;
    for (row = 0; row < rows; row++)
    {
        mat[row] = calloc(cols, sizeof(*mat[row]));
    }
    return mat;
}
static int count_lines(FILE *in_file)
{
    int count = 1;
    char c;
    while ((c = getc(in_file)) != EOF)
    {
        if (c == '\n')
        {
            count++;
        }
    }
    rewind(in_file);
    return count;
}
static int find_dim(FILE *in_file)
{
    int count = 1;
    char c;
    while ((c = getc(in_file)) != '\n')
    {
        if (c == ',')
        {
            count++;
        }
    }
    rewind(in_file);
    return count;
}
static double **read_input_file(char const *input_file_path, int *lines, int *dim)
{
    FILE *in_file = fopen(input_file_path, "r");
    double **vectors = create_matrix(*lines = count_lines(in_file), *dim = find_dim(in_file));
    int i;
    for (i = 0; i < *lines; i++)
    {
        int j;
        for (j = 0; j < *dim; j++)
        {
            fscanf(in_file, "%lf,", &vectors[i][j]);
        }
        fscanf(in_file, "%lf\n", &vectors[i][j]);
    }
    fclose(in_file);
    return vectors;
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
free the allocated memory of mat (mat is array with size rows of arrays)
*/
static void free_matrix(double **matrix, double rows)
{
    int row;
    for (row = 0; row < rows; row++)
    {
        free(matrix[row]);
    }
    free(matrix);
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
    int lines;
    double **input_vectors = read_input_file(input_file_path, &lines, &dim);
    double **centroids = initialize_centroids(k, dim);
    double **clusters_sum = create_matrix(k, dim);
    double *clusters_lens = calloc(k, sizeof(double));
    int i;
    for (i = 0; i < max_iter; i++)
    {
        bool convergence = true;
        int vector_idx;
        int centroid_idx;
        for (vector_idx = 0; vector_idx < lines; vector_idx++)
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
    free_matrix(input_vectors, lines);
    free_matrix(centroids, k);
    free_matrix(clusters_sum, k);
    free(clusters_lens);
}
