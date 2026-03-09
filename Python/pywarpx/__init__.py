# Copyright 2016-2024 The WarpX Community
#
# This file is part of WarpX.
#
# Authors: Andrew Myers, David Grote, Lorenzo Giacomel, Axel Huebl, S. Eric Clark
# License: BSD-3-Clause-LBNL

import os

# Python 3.8+ on Windows: DLL search paths for dependent
# shared libraries
# Refs.:
# - https://github.com/python/cpython/issues/80266
# - https://docs.python.org/3.8/library/os.html#os.add_dll_directory
if os.name == "nt":
    # add anything in the current directory
    pwd = __file__.rsplit(os.sep, 1)[0] + os.sep
    os.add_dll_directory(pwd)
    # add anything in PATH
    paths = os.environ.get("PATH", "")
    for p in paths.split(";"):
        p_abs = os.path.abspath(os.path.expanduser(os.path.expandvars(p)))
        if os.path.exists(p_abs):
            os.add_dll_directory(p_abs)

# Runtime (compiled C extension) imports.  These require numpy, MPI, and the
# WarpX pybind11 shared libraries.  Wrap in try/except so that lightweight
# tooling such as input generation can import ``pywarpx`` (and subpackages
# like ``pywarpx.inputgen``) in environments where the full WarpX stack is not
# installed.
try:
    from ._libwarpx import libwarpx  # noqa
    from .Algo import algo  # noqa
    from .Amr import amr  # noqa
    from .Amrex import amrex  # noqa
    from .Boundary import boundary  # noqa
    from .Collisions import collisions  # noqa
    from .Constants import my_constants  # noqa
    from .Diagnostics import diagnostics, reduced_diagnostics  # noqa
    from .EB2 import eb2  # noqa
    from .Geometry import geometry  # noqa
    from .HybridPICModel import hybridpicmodel, external_vector_potential  # noqa
    from .Interpolation import interpolation  # noqa
    from .Lasers import lasers  # noqa
    from .LoadThirdParty import load_cupy  # noqa
    from .Particles import new_species, particles  # noqa
    from .PSATD import psatd  # noqa
    from .WarpX import warpx  # noqa
    _RUNTIME_AVAILABLE = True
except ImportError:
    libwarpx = None  # type: ignore
    _RUNTIME_AVAILABLE = False

# This is a circular import and must happen after the import of libwarpx
#
# NOTE: importing PICMI pulls in optional dependencies (e.g. periodictable).
# Keep this import tolerant so lightweight tooling (e.g. input generation)
# can import `pywarpx` without requiring the full PICMI dependency chain.
try:
    from . import picmi  # noqa  # isort:skip
except ModuleNotFoundError:
    picmi = None  # type: ignore


# intentionally query the value - only set once sim dimension is known
def __getattr__(name):
    # https://stackoverflow.com/a/57263518/2719194
    if name == "__version__":
        if libwarpx is not None:
            return libwarpx.__version__
        raise AttributeError("pywarpx runtime not available (compiled extensions not installed)")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# TODO
# __doc__ = cxx.__doc__
# __license__ = cxx.__license__
# __author__ = cxx.__author__
