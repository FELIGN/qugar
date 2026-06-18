# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.23)

# Set the project name and language
set(qugar_VERSION "0.5.0")
project(
  qugar 
  VERSION ${qugar_VERSION}
  DESCRIPTION ""
  HOMEPAGE_URL "https://github.com/pantolin/qugar"
  )

include(../cmake/GitConfig.cmake)
include(../cmake/PreventInSourceBuilds.cmake)
include(../cmake/ProjectOptions.cmake)
include(../cmake/Dependencies.cmake)
include(../cmake/Install.cmake)

set(qugar_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(qugar_BINARY_DIR ${PROJECT_BINARY_DIR})

set(qugar_INCLUDE_DIR ${qugar_SOURCE_DIR}/include)
set(qugar_TEST_DIR ${qugar_SOURCE_DIR}/test)
set(qugar_DOC_DIR ${qugar_SOURCE_DIR}/doc)

# set(cpp_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)


git_config()


# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

# Only set the cxx_standard if it is not set by someone else
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS ON)

qugar_setup_config_options()
qugar_setup_options()

qugar_global_options()
qugar_setup_dependencies()

qugar_local_options()

# don't know if this should be set globally from here or not...
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Adding sources
add_subdirectory(qugar)

# Don't even look at tests if we're not top level
if(NOT PROJECT_IS_TOP_LEVEL)
  return()
endif()

# Adding the tests:
include(CTest)

if(BUILD_TESTING)
  # message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
  add_subdirectory(test)
endif()


# If MSVC is being used, and ASAN is enabled, we need to set the debugger environment
# so that it behaves well with MSVC's debugger, and we can run the target from visual studio
if(MSVC)
  get_all_installable_targets(all_targets)
  message("all_targets=${all_targets}")
  set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif()

if(qugar_BUILD_DOC)
  add_subdirectory(doc)
endif()

# # set the startup project for the "play" button in MSVC
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT demo_reparam_gygroid)
if(qugar_WITH_DEMOS)
  add_subdirectory(demo)
endif()

qugar_install_project()