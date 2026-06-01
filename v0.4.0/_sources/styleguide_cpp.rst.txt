.. _developers_styleguide_cpp:

C++ style guide
===============

Formatting
----------

`clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ is used
to format files. A `.clang-format` file is included in the root
directory. Editors can be configured to apply the style using
`clang-format`.

Static code analysis
--------------------

QUGaR has been configured for running
`Clang-Tidy <https://clang.llvm.org/extra/clang-tidy/>`_ and
`Cppcheck <https://cppcheck.sourceforge.io)>`_ at build time.
These static code analysis tools will help you to catch potential bugs, conform to
QUGaR's coding style, and improve the code quality.

Clang-Tidy's configuration is defined in file ``.clang-tidy``, located in the root
directory, while Cppcheck is configured in the file ``cmake/StaticAnalyzers.cmake``.
Make sure to enable them when developing new features by setting
the variable ``qugar_DEVELOPER_MODE`` to ``ON`` when invoking CMake.
This variable gets activated by default when building in ``Debug`` mode.

ISO C++ standard
----------------

QUGaR relies on C++20 standard.


Naming conventions
------------------

Class names
^^^^^^^^^^^
Use camel caps for class names:

.. code-block:: c++

    class FooBar
    {
      ...
    };

Function names
^^^^^^^^^^^^^^

Use lower-case for function names and underscore to separate words:

.. code-block:: c++

    foo();
    bar();
    foo_bar(...);

.. Functions returning a value should be given the name of that value,
.. for example:

.. .. code-block:: c++

..     class Array:
..     {
..     public:

..       /// Return size of array (number of entries)
..       std::size_t size() const;

..     };

.. In the above example, the function should be named ``size`` rather
.. than ``get_size``. On the other hand, a function not returning a value
.. but rather taking a variable (by reference) and assigning a value to
.. it, should use the ``get_foo`` naming scheme, for example:

.. .. code-block:: c++

..     class Parameters:
..     {
..     public:

..       /// Retrieve all parameter keys
..       void get_parameter_keys(std::vector<std::string>& parameter_keys) const;

..     };


Variable names
^^^^^^^^^^^^^^

Use lower-case for variable names and underscore to separate words:

.. code-block:: c++

    Foo foo;
    Bar bar;
    FooBar foo_bar;

Use an underscore prefix for class member variables:

.. code-block:: c++

    class Foo:
    {
      int bar_;
    };

Enum variables and constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum variables should be lower-case with underscore to separate words:

.. code-block:: c++

    enum Type {foo, bar, foo_bar};

We try to avoid using ``#define`` to define constants, but when
necessary constants should be capitalized:

.. code-block:: c++

    #define FOO 3.14159265358979

File names
^^^^^^^^^^

Use lower-case and underscores to separate words for file names. Header files should have the suffix
``.hpp`` and implementation files should have the suffix ``.cpp``:

.. code-block:: c++

    foo_bar.hpp
    foo_bar.cpp


Miscellaneous
-------------

Comments
^^^^^^^^

Capitalize the first letter of a comment and use punctuation. Here's an example:

.. code-block:: c++

    // Check if connectivity has already been computed.
    if (!connectivity.empty()) {
      return;
    }

    // Invalidate ordering.
    mesh._ordered = false;

    // Compute entities if they don't exist.
    if (topology.size(d0) == 0) {
      compute_entities(mesh, d0);
    }
    if (topology.size(d1) == 0) {
      compute_entities(mesh, d1);
    }

    // Check if connectivity still needs to be computed.
    if (!connectivity.empty()) {
      return;
    }

    ...

Always use ``//`` for comments and ``//!`` for documentation. Never
use ``/* foo */``, not even for comments that runs over multiple
lines.


Header file layout
^^^^^^^^^^^^^^^^^^

Header files should follow the below template:

.. code-block:: c++

    // --------------------------------------------------------------------------
    //
    // Copyright (C) 2025-present by Pablo Antolin
    //
    // This file is part of the QUGaR library.
    //
    // SPDX-License-Identifier:    MIT
    //
    // --------------------------------------------------------------------------

    #ifndef QUGAR_LIBRARY_FOO_HPP
    #define QUGAR_LIBRARY_FOO_HPP

    //! @file foo.hpp
    //! @author Pablo Antolin (pablo.antolin@epfl.ch)
    //! @brief Definition of Foo class.
        //! @date 2025-01-21
    //!
    //! @copyright Copyright (c) 2025-present


    namespace qugar {

    //! Documentation of class

    class Foo
    {
    public:

      ...

    private:

      ...

    };

    } // namespace qugar 

    #endif // QUGAR_LIBRARY_FOO_HPP

Implementation file layout
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implementation files should follow the below template:

.. code-block:: c++

    // --------------------------------------------------------------------------
    //
    // Copyright (C) 2025-present by Pablo Antolin
    //
    // This file is part of the QUGaR library.
    //
    // SPDX-License-Identifier:    MIT
    //
    // --------------------------------------------------------------------------
  
    //! @file foo.cpp
    //! @author Pablo Antolin (pablo.antolin@epfl.ch)
    //! @brief Implementation of Foo class.
        //! @date 2025-01-21
    //!
    //! @copyright Copyright (c) 2025-present

    #include <qugar/foo.h>

    namespace qugar {

    Foo::Foo() : // variable initialization here
    {
      ...
    }

    Foo::~Foo()
    {
      // Do nothing
    }

    // Template instantiations (if needed)

    } // namespace qugar


Including header files and using forward declarations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only include the portions of QUGaR you are actually using.

Include all the header files that are needed, but as few as possible.
`Avoid using forward declarations <https://google.github.io/styleguide/cppguide.html#Forward_Declarations>`_ whenever possible (in header files).
Using them will definitely speed up compilation times, but they can hide
dependencies and makes it difficult for automatic tooling to discover the module defining the symbol.

Explicit constructors
^^^^^^^^^^^^^^^^^^^^^

Make all one argument constructors (except copy constructors)
explicit:

.. code-block:: c++

    class Foo
    {
      explicit Foo(std::size_t i);
    };

Virtual functions
^^^^^^^^^^^^^^^^^

Always declare inherited virtual functions as ``virtual`` in the
subclasses.  This makes it easier to spot which functions are virtual.
Use the ``final`` keyword to indicate that a function should not be
overridden.

.. code-block:: c++

    class Foo
    {
      virtual void foo();
      virtual void bar() = 0;
    };

    class Bar : public Foo
    {
      virtual void foo() final;
      virtual void bar() final;
    };

Use of libraries
----------------

Prefer C++ strings and streams over old C-style ``char*``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``std::string`` instead of ``const char*`` and use
``std::istream`` and ``std::ostream`` instead of ``FILE``. Avoid
``printf``, ``sprintf`` and other C functions.

There are some exceptions to this rule where we need to use old
C-style function calls. One such exception is handling of command-line
arguments (``char* argv[]``).

Avoid plain pointers
^^^^^^^^^^^^^^^^^^^^

Use C++11 smart pointer and avoid plain pointers.
