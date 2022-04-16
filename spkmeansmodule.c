#include <Python.h>
#include "spkmeans.c"
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