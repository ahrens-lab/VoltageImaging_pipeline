#!/usr/bin/env python

# compile with python setup.py build_ext -i

from setuptools.extension import Extension
from setuptools import setup

setup(ext_modules=[Extension('imfc',
                             include_dirs=['/usr/local/include'],
                             library_dirs=['/usr/local/lib'],
                             sources=['imfc/imfc/imfc.cpp'])])
