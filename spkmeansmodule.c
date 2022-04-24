#include <Python.h>
#include "spkmeans.h"
#define PY_SSIZE_T_CLEAN
#define INPUT_FILE "my_input.txt"

/*
 * API functions
 */
static PyObject *algorithm(PyObject *self, PyObject *args)
{
    int k;
    const char *goal;
    if (!PyArg_ParseTuple(args, "is", &k, &goal))
    {
        return NULL;
    }
    algorithm(goal, INPUT_FILE, k, write_output)
    Py_RETURN_NONE;
}
/*
 * A macro to help us with defining the methods
 */
#define FUNC(_flag, _name, _docstring)                           \
    {                                                            \
#_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) \
    }
static PyMethodDef _methods[] = {
    FUNC(METH_VARARGS, algorithm, "mykmeanssp"),
    {NULL, NULL, 0, NULL} /* sentinel */
};
static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods};
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}