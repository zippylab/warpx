# Copyright 2017-2022 The WarpX Community
#
# This file is part of WarpX. It defines the wrapper functions that directly
# call the underlying compiled routines through pybind11.
#
# NOTE: We will reduce the libwarpx.py level of abstraction eventually!
# Please add new functionality directly to pybind11-bound modules
# and call them via sim.extension.libwarpx_so. ... and sim.extension.Config and
# sim.extension.warpx. ... from user code.
#
# Authors: Axel Huebl, Andrew Myers, David Grote, Remi Lehe, Weiqun Zhang
#
# License: BSD-3-Clause-LBNL

import atexit
import os
import sys
import glob

# Ensure the local build's shared libraries (e.g., libamrex_2d.so) are discoverable.
# This is needed when running directly from a source tree without installing
# pywarpx/amrex into the active environment.
_this_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_local_build_root = os.path.join(_this_repo_root, "pywarpx")

# 1) Prefer running against an installed Python package layout in `pywarpx/site-packages`.
#    This contains the compiled `warpx_pybind_*.so` modules and the `amrex` Python package.
_site_pkgs = os.path.join(_local_build_root, "site-packages")
if os.path.isdir(_site_pkgs) and _site_pkgs not in sys.path:
    sys.path.insert(0, _site_pkgs)

# If we're running from a source tree, `pywarpx` is this package (Python/pywarpx).
# The built extension modules live in `pywarpx/site-packages/pywarpx`, so extend
# the package search path to find e.g. `pywarpx.warpx_pybind_2d`.
try:
    import pywarpx as _pywarpx_pkg

    _pywarpx_ext_pkg = os.path.join(_site_pkgs, "pywarpx")
    if os.path.isdir(_pywarpx_ext_pkg) and _pywarpx_ext_pkg not in _pywarpx_pkg.__path__:
        _pywarpx_pkg.__path__.append(_pywarpx_ext_pkg)
except Exception:
    pass

# 2) Ensure the local build's shared libraries (e.g., libamrex_2d.so) are discoverable.
#    Note: changing LD_LIBRARY_PATH *after process start* might not affect the dynamic
#    loader on all systems.
if os.path.isdir(_local_build_root) and _local_build_root not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _local_build_root + (
        ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
    )

# Additionally, pre-load key shared libraries by absolute path so that downstream
# pybind modules can resolve their DT_NEEDED entries even if LD_LIBRARY_PATH isn't
# honored after process start.
try:
    import ctypes

    # AMReX libraries built alongside WarpX
    for _cand in ("libamrex_1d.so", "libamrex_2d.so", "libamrex_3d.so", "libamrex.so"):
        _p = os.path.join(_local_build_root, _cand)
        if os.path.exists(_p):
            ctypes.CDLL(_p)

    # ADIOS2: WarpX pybind modules may link against libadios2_cxx11.so.
    # Prefer an already-configured runtime (e.g. module environment) via LD_LIBRARY_PATH.
    # As a best-effort fallback, try to locate ADIOS2 libs in common installs.
    try:
        _adios2_cands = []

        # 1) If ADIOS2 python package is installed, its shared libs live next to it.
        try:
            import adios2  # type: ignore

            _adios2_cands.append(os.path.realpath(os.path.dirname(adios2.__file__)))
        except Exception:
            pass

        # 2) Known spack-style install roots on ALCF systems (e.g. Aurora)
        for _root in (
            "/opt/aurora",
            "/lus/flare/projects/catalyst/world_shared",
        ):
            if os.path.isdir(_root):
                # Keep this search shallow-ish by walking only a couple levels of directories.
                for _sub in os.listdir(_root):
                    _p = os.path.join(_root, _sub)
                    if os.path.isdir(_p) and "adios2" in _sub:
                        _adios2_cands.append(_p)

        # If any candidate is actually a libdir, add it.
        for _d in list(dict.fromkeys(_adios2_cands)):
            if os.path.isdir(_d) and _d not in os.environ.get("LD_LIBRARY_PATH", ""):
                os.environ["LD_LIBRARY_PATH"] = _d + (
                    ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
                )

        # Try load from LD_LIBRARY_PATH via soname first.
        try:
            ctypes.CDLL("libadios2_cxx11.so.2.10")
        except Exception:
            # Try a direct filesystem search for a matching soname and load by absolute path.
            _found = []
            for _base in ("/opt/aurora",):
                if os.path.isdir(_base):
                    for _p in glob.glob(os.path.join(_base, "**", "libadios2_cxx11.so.2.10"), recursive=True)[:50]:
                        _found.append(_p)
            for _p in _found:
                try:
                    ctypes.CDLL(_p)
                    break
                except Exception:
                    continue
    except Exception:
        pass
except Exception:
    # Best-effort only.
    pass

from .Geometry import geometry


class LibWarpX:
    """This class manages the WarpX classes, part of the Python module from the compiled C++ code.
    It will only load the library when it is referenced, and this can only be done after
    the geometry is defined so that the version of the library that is needed can be determined.
    Once loaded, all the settings of function call interfaces are set up.
    """

    def __init__(self):
        # Track whether amrex and warpx have been initialized
        self.initialized = False
        atexit.register(self.finalize)

        # set once libwarpx_so is loaded
        self.__version__ = None

    def __getattr__(self, attribute):
        if attribute == "libwarpx_so":
            # If the 'libwarpx_so' is referenced, load it.
            # Once loaded, it gets added to the dictionary so this code won't be called again.
            self.load_library()
            return self.__dict__[attribute]
        elif attribute == "warpx":
            # A `warpx` attribute has not yet been assigned, so `initialize_warpx` has not been called.
            raise AttributeError(
                "Trying to access libwarpx.warpx before initialize_warpx has been called!"
            )
        else:
            # For any other attribute, call the built-in routine - this should always
            # return an AttributeError.
            return self.__getattribute__(attribute)

    def _get_package_root(self):
        """
        Get the path to the installation location (where libwarpx.so would be installed).
        """
        cur = os.path.abspath(__file__)
        while True:
            name = os.path.basename(cur)
            if name == "pywarpx":
                return cur
            elif not name:
                return ""
            cur = os.path.dirname(cur)

    def load_library(self):
        if "libwarpx_so" in self.__dict__:
            raise RuntimeError(
                "Invalid attempt to load the pybind11 bindings library multiple times. "
                "Note that multiple AMReX/WarpX geometries cannot be loaded yet into the same Python process. "
                "Please write separate scripts for each geometry."
            )

        # --- Use geometry to determine whether to import the 1D, 2D, 3D or RZ version.
        # --- The geometry must be setup before the lib warpx shared object can be loaded.
        try:
            _dims = str(geometry.dims)
        except AttributeError:
            raise Exception(
                "The shared object could not be loaded. The geometry must be setup before the WarpX pybind11 module can be accessed. The geometry determines which version of the shared object to load."
            )

        if _dims == "RZ":
            self.geometry_dim = "rz"
        elif _dims == "1" or _dims == "2" or _dims == "3":
            self.geometry_dim = "%dd" % int(_dims)
        else:
            raise Exception("Undefined geometry %d" % _dims)

        try:
            if self.geometry_dim == "1d":
                import amrex.space1d as amr

                self.amr = amr
                from . import warpx_pybind_1d as cxx_1d

                self.libwarpx_so = cxx_1d
                self.dim = 1
            elif self.geometry_dim == "2d":
                import amrex.space2d as amr

                self.amr = amr
                from . import warpx_pybind_2d as cxx_2d

                self.libwarpx_so = cxx_2d
                self.dim = 2
            elif self.geometry_dim == "rz":
                import amrex.space2d as amr

                self.amr = amr
                from . import warpx_pybind_rz as cxx_rz

                self.libwarpx_so = cxx_rz
                self.dim = 2
            elif self.geometry_dim == "3d":
                import amrex.space3d as amr

                self.amr = amr
                from . import warpx_pybind_3d as cxx_3d

                self.libwarpx_so = cxx_3d
                self.dim = 3

            self.Config = self.libwarpx_so.Config
        except ImportError:
            raise Exception(
                f"Dimensionality '{self.geometry_dim}' was not compiled in this Python install. Please recompile with -DWarpX_DIMS={_dims}"
            )

        self.__version__ = self.libwarpx_so.__version__

        # Extend pybind11 types in libwarpx_so (and pyAMReX) with pure Python
        from .extensions.MultiFab import register_warpx_MultiFab_extension
        from .extensions.MultiFabRegister import (
            register_warpx_MultiFabRegister_extension,
        )
        from .extensions.MultiParticleContainer import (
            register_warpx_MultiParticleContainer_extension,
        )
        from .extensions.WarpXParticleContainer import (
            register_warpx_WarpXParticleContainer_extension,
        )

        register_warpx_MultiFab_extension(self.amr)
        register_warpx_MultiFabRegister_extension(self.libwarpx_so)
        register_warpx_MultiParticleContainer_extension(self.libwarpx_so)
        register_warpx_WarpXParticleContainer_extension(self.libwarpx_so)

    def amrex_init(self, argv, mpi_comm=None):
        if mpi_comm is None:  # or MPI is None:
            self.libwarpx_so.amrex_init(argv)
        else:
            raise Exception("mpi_comm argument not yet supported")

    def initialize(self, argv=None, mpi_comm=None):
        """
        Initialize WarpX and AMReX. Must be called before doing anything else.
        """
        if argv is None:
            argv = sys.argv
        self.amrex_init(argv, mpi_comm)
        self.warpx = self.libwarpx_so.get_instance()
        self.warpx.initialize_data()
        self.libwarpx_so.execute_python_callback("afterinit")
        self.libwarpx_so.execute_python_callback("particleloader")

        self.initialized = True

    def finalize(self, finalize_mpi=1):
        """
        Call finalize for WarpX and AMReX. Registered to run at program exit.
        """
        # TODO: simplify, part of pyAMReX already
        if self.initialized:
            del self.warpx
            # The call to warpx_finalize causes a crash - don't know why
            # self.libwarpx_so.warpx_finalize()
            self.libwarpx_so.amrex_finalize()

            from pywarpx import callbacks

            callbacks.clear_all()


libwarpx = LibWarpX()
