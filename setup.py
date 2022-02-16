from setuptools import setup, Extension

setup(name='mykmeanssp',
      version='1.0',
      description='kmeans algorithem',
      ext_modules=[Extension('mykmeanssp', sources=['kmeans.c'])])
