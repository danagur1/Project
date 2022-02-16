#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define DEFAULTMAXITER 200
/*
 * Helper functions implementation
 */
typedef enum
{
    false,
    true
} bool;
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
    return count;
}
double **read_input_file(FILE *in_file, int lines, int dim)
{
    double **vectors;
    int i, j;
    vectors = malloc(lines * sizeof(*vectors));
    for (i = 0; i < lines; i++)
    {
        *(vectors + i) = malloc(dim * sizeof(**(vectors + i)));
    }
    fseek(in_file, 0, SEEK_SET);
    for (i = 0; i < lines; i++)
    {
        for (j = 0; j < dim; j++)
        {
            fscanf(in_file, j != dim - 1 ? "%lf," : "%lf\n", *(vectors + i) + j);
        }
    }
    return vectors;
}
static double **initialize_centroids(double **vectors, int k, int dim)
{
    FILE *in_file = fopen("first_centroids.txt", "r");
    double **centroids;
    int i, j;
    centroids = malloc(k * sizeof(*centroids));
    for (i = 0; i < k; i++)
    {
        *(centroids + i) = malloc(dim * sizeof(**(centroids + i)));
    }
    for (i = 0; i < k; i++)
    {
        for (j = 0; j < dim; j++)
        {
            fscanf(in_file, j != dim - 1 ? "%lf," : "%lf\n", *(centroids + i) + j);
        }
    }
    return centroids;
}
static double **initialize_clusters(int k, int dim)
{
    double **clusters_sum = malloc(sizeof(*clusters_sum) * k);
    int i;
    for (i = 0; i < k; i++)
    {
        *(clusters_sum + i) = calloc(dim, sizeof(double));
    }
    return clusters_sum;
}
static double euclidean_norm(double *vector, int dim)
{
    double square_sum = 0;
    int i;
    for (i = 0; i < dim; i++)
    {
        square_sum += pow(*(vector + i), 2);
    }
    return sqrt(square_sum);
}
static void vectors_operators(double *new_vector, double *a, double *b, char op, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
    {
        switch (op)
        {
        case '+':
            *(new_vector + i) = *(a + i) + *(b + i);
            break;
        case '-':
            *(new_vector + i) = *(a + i) - *(b + i);
            break;
        default:
            break;
        }
    }
    return;
}
static double dist(double *a, double *b, int dim)
{
    double *minus = malloc(dim * sizeof(*minus));
    vectors_operators(minus, a, b, '-', dim);
    double result = euclidean_norm(minus, dim);
    free(minus);
    return result;
}
static int find_best_cluster(int k, double **centroids, double *vector, int dim)
{
    int i, min_cluster = 0;
    double min_dist = dist(*centroids, vector, dim);
    for (i = 1; i < k; i++)
    {
        double curr_dist = dist(vector, *(centroids + i), dim);
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
    double *result = malloc(sizeof(double) * dim);
    int i;
    for (i = 0; i < dim; i++)
        *(result + i) = *(a + i) / d;
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
            double value = *(*(centroids + i) + j);
            fprintf(out_file, j != dim - 1 ? "%.4f," : "%.4f\n", value);
        }
    }
    fclose(out_file);
}
static void free_mem(double **a, double **b, double **c, double *d, int k, int lines)
{
    int i;
    for (i = 0; i < lines; i++)
    {
        if (i < k)
        {
            free(*(b + i));
            free(*(c + i));
        }
        free(*(a + i));
    }
    free(a);
    free(b);
    free(c);
    free(d);
}
static void clusters_reset(double **clusters_sum, double *clusters_lens, int k, int dim)
{
    int i, j;
    for (i = 0; i < k; i++)
    {
        *(clusters_lens + i) = 0;
        for (j = 0; j < dim; j++)
        {
            *(*(clusters_sum + i) + j) = 0;
        }
    }
}
static int kmeans(int k, int max_iter, double epsilon, const char *input_file_path, const char *output_file_path)
{
    FILE *in_file = fopen(input_file_path, "r");
    int dim = find_dim(in_file);
    int lines = count_lines(in_file);
    double **input_vectors = read_input_file(in_file, lines, dim);
    double **centroids = initialize_centroids(input_vectors, k, dim);
    double **clusters_sum = initialize_clusters(k, dim);
    double *clusters_lens = calloc(k, sizeof(double));
    bool convergence = false;
    double *vector, *new_centroid;
    int best_cluster, vector_idx, centroid_idx, i = 0;
    while ((i != max_iter) && (!convergence))
    {
        i++;
        clusters_reset(clusters_sum, clusters_lens, k, dim);
        for (vector_idx = 0; vector_idx < lines; vector_idx++)
        {
            vector = *(input_vectors + vector_idx);
            best_cluster = find_best_cluster(k, centroids, vector, dim);
            vectors_operators(*(clusters_sum + best_cluster), clusters_sum[best_cluster], vector, '+', dim);
            (*(clusters_lens + best_cluster))++;
        }
        convergence = true;
        for (centroid_idx = 0; centroid_idx < k; centroid_idx++)
        {
            new_centroid = divide(*(clusters_sum + centroid_idx), *(clusters_lens + centroid_idx), dim);
            if (fabs(euclidean_norm(*(centroids + centroid_idx), dim) - euclidean_norm(new_centroid, dim)) > epsilon)
            {
                convergence = false;
            }
            free(*(centroids + centroid_idx));
            *(centroids + centroid_idx) = new_centroid;
        }
    }
    write_output(output_file_path, centroids, k, dim);
    free_mem(input_vectors, centroids, clusters_sum, clusters_lens, k, lines);
    return 0;
}
/*
 * API functions
 */
static PyObject* fit(PyObject *self, PyObject *args)
{
    int k;
    int max_iter;
    double epsilon;
    if(!PyArg_ParseTuple(args, "iid", &k, &max_iter, &epsilon)) {
        return NULL; 
    }
    kmeans(k, max_iter, epsilon, "input_vectors.txt", "final_centroids.txt");
    Py_RETURN_NONE;
}
/*
 * A macro to help us with defining the methods
 * Compare with: {"f1", (PyCFunction)f1, METH_NOARGS, PyDoc_STR("No input parameters")}
*/
#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }
static PyMethodDef _methods[] = {
    FUNC(METH_VARARGS, fit, "mykmeanssp"),
    {NULL, NULL, 0, NULL}   /* sentinel */
};
static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}
