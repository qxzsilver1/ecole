cmake_minimum_required(VERSION 3.14)


project(
	Ecole
	VERSION 0.2.0
	LANGUAGES CXX
	DESCRIPTION "Extensible Combinatorial Optimization Learning Environments"
)
set(Ecole_AUTHORS "Antoine Prouvost")

# Set default parameters. Assumes Ecole user,
include(cmake/DefaultSettings.cmake)

# Define a target Ecole::warnings with all compiler warnings.
include(cmake/CompilerWarnings.cmake)

# Define a target Ecole::sanitizers with enabled sanitizers.
include(cmake/Sanitizers.cmake)

# Define a target Ecole::coverage with coverage options.
include(cmake/Coverage.cmake)

# Activate the Conan CMake wrapper (download if necessary).
include(cmake/Conan.cmake)

# Find Python and create virtual environment for Ecole
include(cmake/Python.cmake)
include(cmake/FindNumPy.cmake)
include(cmake/PythonVenv.cmake)

# Ecole libraries
add_subdirectory(libecole)
add_subdirectory(python)

# Documentation with Doxygen and Sphinx
add_subdirectory(docs EXCLUDE_FROM_ALL)
