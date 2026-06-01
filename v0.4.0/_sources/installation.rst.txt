.. QUGaR installation docs

Installation
============

Right now, QUGaR can only be installed from its sources.
In the near future we will provide a conda package to ease the process.
Alternatively, you can use the a Docker image, as described below.

Source
------

QUGaR is composed by two main components: the C++ core and an (optional)  Python interface.


Dependencies
^^^^^^^^^^^^

C++ core
********

The C++ core can be installed without Python as a dependency.

.. rubric:: Required

- C++ compiler (supporting the C++20 standard)
- `CMake <https://cmake.org>`_
- `Algoim <https://algoim.github.io>`_ (automatically downloaded and installed during the CMake configuration process)

.. rubric:: Optional

- `The LAPACKE C Interface to LAPACK <https://www.netlib.org/lapack/lapacke.html>`_ (this is an optional dependency, but strongly recommended. If not found, built-in eigenvalue solvers will be used)
- `Catch2 <https://github.com/catchorg/Catch2>`_ (required for testing. If not found, it is automatically downloaded and installed during the CMake configuration process)

.. rubric:: Optional static analysis tools for developers

- `Cppcheck <http://cppcheck.sourceforge.net/>`_
- `Clang-Tidy <https://clang.llvm.org/extra/clang-tidy/>`_


Python interface
****************

Below are additional requirements for the Python interface to the C++
core. Except for FEniCSx, all dependencies can be automatically installed
during the Python package installation process.

.. rubric:: Required

- `Python3 <https://www.python.org/downloads/>`_
- `nanobind <https://pypi.org/project/nanobind/>`_
- `scikit-build-core[pyproject] <https://pypi.org/project/scikit-build-core/>`_
- `NumPy <https://pypi.org/project/numpy/>`_

.. rubric:: Optional

- `FEniCSx <https://fenicsproject.org>`_ v0.10.0 (``fenics-dolfinx``, ``fenics-basix``, ``fenics-ffcx`` all ``>=0.10.0,<0.11.0``; ``fenics-ufl`` ``>=2025.2.0,<2025.3.0``). See ``python/pyproject.toml`` for the authoritative pin list.
- `PyVista <https://pypi.org/project/pyvista>`__

.. rubric:: Optional for testing

- `pytest <https://pypi.org/project/pytest/>`_

.. rubric:: Optional for linting

- `mypy <https://pypi.org/project/mypy/>`_
- `ruff <https://pypi.org/project/ruff/>`_

Documentation generation
************************
Except for Doxygen, all the below dependencies can be automatically installed
during the Python interface installation process, or using the ``doc/requirements.txt`` file.

.. rubric:: Required

- `Doxygen <https://www.doxygen.nl>`_
- `sphinx <https://pypi.org/project/Sphinx/>`_
- `sphinx-rtd-dark-mode <https://pypi.org/project/sphinx-rtd-dark-mode/>`_
- `breathe <https://pypi.org/project/breathe/>`_
- `jupytext <https://pypi.org/project/jupytext/>`_
- `markdown <https://pypi.org/project/Markdown/>`_
- `myst_parser <https://pypi.org/project/myst-parser/>`_

Building and installing
^^^^^^^^^^^^^^^^^^^^^^^

First clone the repository and go inside::

    git clone https://github.com/pantolin/qugar.git
    cd qugar

C++ core
********
To build and install the C++ library, from the ``qugar/`` root directory run::

    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build -S cpp/
    cmake --build build --parallel 4
    cmake --install build

The following options can be appended to the first ``cmake`` command:

- ``-DBUILD_TESTING=ON``: to build C++ tests.
- ``-Dqugar_WITH_DEMOS=ON``: for building some basic demos.
- ``-Dqugar_DEVELOPER_MODE=ON``: for activating multiple compiler options, including sanitizers. This option is automatically activated when compiling in Debug mode.
- ``-DLAPACKE_DIR=<lapacke-path>``: to provide the specific path to LAPACKE. Some algorithms can benefit from the use of `LAPACKE <https://www.netlib.org/lapack/lapacke.html>`_ for solving eigenvalue problems. For instance, when finding roots of polynomials. If LAPACKE is not found during the configuration the message ``LAPACKE not found. Algoim's eigenvalue solvers will be used.`` is displayed.
- ``-DCMAKE_INSTALL_PREFIX=<my-install-path>``: to provide a specific installation path.
- ``-DCMAKE_CXX_COMPILER=<my-c++-commpiler-path>``: to provide a specific C++ compiler.
- etc.

Optionally, it is possible to use CMakePresets. For doing so create ``CMakeUserPresets.json`` files both in the root folder and in the ``cpp/`` and invoke ``cmake`` as::

    cmake --preset=<preset-name> -B build -S cpp/

Examples of both ``CMakeUserPresets.json`` files can be found in the root folder and the ``cpp/`` folders under the names ``CMakeUserPresets.json.template``.

Python
******

QUGaR's Python interface requires the C++ library. So before installing the Python interface, make sure you have the C++ library installed.

To build and install the Python interface, under the ``qugar/`` root directory run::

    python3 -m pip -v install -r ./python/build-requirements.txt
    python3 -m pip -v install --no-build-isolation ./python -U

Optional dependencies, including the documentation dependencies (see below), can be installed by replacing the last line with::

    python3 -m pip -v install --no-build-isolation ./python[all]

QUGaR's Python interface is designed to interact nicely with `FEniCSx <https://fenicsproject.org>`_. Check the Demos page examples.
So, if you want to solve your PDEs using unfitted domains through FEniCSx, make sure to install `DOLFINx <https://github.com/FEniCS/dolfinx>`_ v0.10.0 (and the matching versions of basix, ffcx and ufl listed above).


However, it is also possible to use QUGaR's Python interface without DOLFINx (check the demos page for examples).

QUGaR's Python interface provides some extra visualization features through the `PyVista <https://pyvista.org>`__ library. To enable such features, install `pyvista` (for instance, from `PyPI <https://pypi.org/project/pyvista/>`__ or `conda-forge <https://anaconda.org/conda-forge/pyvista>`__).


Documentation
*************

Once the QUGaR's Python interface has been installed (and the C++ library), the documentation can be built and installed as follows::

    cd doc
    python3 -m pip -v install -r requirements.txt
    python3 build_docs.py

The documentation will be generated in the ``pages/`` directory.

Note that Doxygen must be installed in your system to build the documentation.

Documentation for the `main` branch can be found `here <https://pantolin.github.io/qugar/main/index.html>`_.

Docker
------

It is also possible to use QUGaR from a Docker container.
The docker file may be built and run from the ``qugar/`` root directory as::

    docker build -f docker/Dockerfile -t qugar .
    docker run -it -v $(pwd):/root/shared -w /root/shared qugar bash -i
