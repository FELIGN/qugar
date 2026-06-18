#!/usr/bin/env bash
set -euo pipefail

# QUGaR Complete Setup and Installation Script
# This script:
#   1. Creates a conda environment with all required tools
#   2. Builds and installs QUGaR library and Python interface
# Supports: macOS (Intel/Apple Silicon), Linux (x86_64/ARM), Windows (via Git Bash/WSL)

ENV_NAME="${QUGAR_ENV_NAME:-qugar-env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_LAPACKE="${INSTALL_LAPACKE:-true}"
INSTALL_DOLFINX="${INSTALL_DOLFINX:-true}"
USE_CONDA_COMPILERS="${USE_CONDA_COMPILERS:-false}"
SKIP_ENV_SETUP="${SKIP_ENV_SETUP:-false}"

# Initialize CMAKE_PREFIX_PATH to avoid unbound variable errors in conda deactivation scripts
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Darwin*)
            PLATFORM="macos"
            ARCH="$(uname -m)"
            ;;
        Linux*)
            PLATFORM="linux"
            ARCH="$(uname -m)"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            PLATFORM="windows"
            ARCH="x86_64"
            ;;
        *)
            log_error "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac
    log_info "Detected platform: ${PLATFORM} (${ARCH})"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check conda
    if ! command -v conda &> /dev/null; then
        log_error "conda not found. Please install Miniconda or Anaconda."
        exit 1
    fi
    log_success "conda found: $(conda --version)"
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_warn "git not found. Will install via conda."
    else
        log_success "git found: $(git --version)"
    fi
}

# Safe conda activate wrapper to handle unbound variables in conda scripts
safe_conda_activate() {
    local env_name="$1"
    # Temporarily disable unbound variable checking for conda operations
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    set -u
}

# Check if environment already exists
check_existing_env() {
    if conda env list | grep -q "^${ENV_NAME}\s"; then
        log_warn "Environment '${ENV_NAME}' already exists."
        if [[ "${SKIP_ENV_SETUP}" != "true" ]]; then
            echo "Options:"
            echo "  [r] Remove and recreate environment"
            echo "  [u] Update environment with current configuration"
            echo "  [s] Skip (use existing environment)"
            read -p "Choose option (r/u/s, default: s): " -n 1 -r
            echo
            case $REPLY in
                [Rr]*)
                    log_info "Removing existing environment..."
                    conda env remove -n "${ENV_NAME}" -y
                    ;;
                [Uu]*)
                    log_info "Updating environment with current configuration..."
                    safe_conda_activate "${ENV_NAME}"
                    install_conda_packages
                    return 0
                    ;;
                *)
                    log_info "Using existing environment. Activating..."
                    safe_conda_activate "${ENV_NAME}"
                    return 0
                    ;;
            esac
        else
            log_info "Using existing environment. Activating..."
            safe_conda_activate "${ENV_NAME}"
            return 0
        fi
    fi
    return 1
}

# Build package list based on configuration flags
# Sets global PACKAGES array (bash 3.2 compatible)
build_package_list() {
    PACKAGES=()
    
    # Base packages - always included
    PACKAGES+=("python=${PYTHON_VERSION}")
    PACKAGES+=("cmake>=3.20")
    PACKAGES+=("ninja")
    PACKAGES+=("git")
    
    # Add LAPACKE if requested
    if [[ "${INSTALL_LAPACKE}" == "true" ]]; then
        PACKAGES+=("liblapacke")
        PACKAGES+=("libtmglib")
    fi
    
    # Add DOLFINx packages if requested (platform-specific)
    if [[ "${INSTALL_DOLFINX}" == "true" ]]; then
        PACKAGES+=("fenics-dolfinx=0.11.0")
        PACKAGES+=("pyvista")
        PACKAGES+=("scipy")
        case "${PLATFORM}" in
            macos|linux)
                PACKAGES+=("mpich")
                ;;
            windows)
                log_warn "Note: PETSc and petsc4py are not available on Windows conda packages (beta testing)"
                PACKAGES+=("pyamg")
                ;;
            *)
                log_warn "Unknown platform ${PLATFORM}, using Linux/macOS defaults..."
                PACKAGES+=("mpich")
                ;;
        esac
    fi
    
    # Add conda compilers if requested (Linux/Windows only, macOS uses system compilers)
    if [[ "${USE_CONDA_COMPILERS}" == "true" ]]; then
        case "${PLATFORM}" in
            linux|windows)
                PACKAGES+=("cxx-compiler")
                PACKAGES+=("c-compiler")
                ;;
            macos)
                log_info "Skipping conda compilers on macOS (using system compilers)"
                ;;
        esac
    fi
}

# Install conda packages directly based on flags and platform
install_conda_packages() {
    log_info "Installing conda packages based on configuration..."
    
    build_package_list
    
    # Install all packages in one command
    if [[ ${#PACKAGES[@]} -gt 0 ]]; then
        log_info "Installing packages: ${PACKAGES[*]}"
        conda install -c conda-forge "${PACKAGES[@]}" -y
    else
        log_warn "No packages to install"
    fi
}

# Create conda environment
create_conda_env() {
    log_info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    
    # Build package list
    build_package_list
    
    # Create environment with all packages in one command
    if [[ ${#PACKAGES[@]} -gt 0 ]]; then
        log_info "Creating environment with packages: ${PACKAGES[*]}"
        conda create -n "${ENV_NAME}" -c conda-forge "${PACKAGES[@]}" -y
    else
        log_warn "No packages specified, creating empty environment..."
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
    
    log_info "Activating environment..."
    safe_conda_activate "${ENV_NAME}"
    
    log_success "Environment created and activated"
}

activate_conda_env() {
    log_info "Activating conda environment '${ENV_NAME}'..."
    safe_conda_activate "${ENV_NAME}"
    log_success "Environment activated"
}

# Verify packages are installed
verify_packages() {
    log_info "Verifying packages..."
    
    local missing_packages=()
    
    # Check build tools
    if [[ ! -f "${CONDA_PREFIX}/bin/cmake" ]]; then
        missing_packages+=("cmake")
    fi
    if [[ ! -f "${CONDA_PREFIX}/bin/ninja" ]]; then
        missing_packages+=("ninja")
    fi
    if [[ ! -f "${CONDA_PREFIX}/bin/git" ]]; then
        missing_packages+=("git")
    fi
    
    # Check LAPACKE if requested
    if [[ "${INSTALL_LAPACKE}" == "true" ]]; then
        if [[ ! -f "${CONDA_PREFIX}/include/lapacke.h" ]] && ! conda list | grep -q "lapacke"; then
            missing_packages+=("liblapacke")
        fi
    fi
    
    # Check DOLFINx if requested
    if [[ "${INSTALL_DOLFINX}" == "true" ]]; then
        if ! conda list | grep -q "fenics-dolfinx"; then
            missing_packages+=("fenics-dolfinx")
        fi
    fi
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_warn "Some packages are missing: ${missing_packages[*]}"
        log_info "Installing missing packages..."
        install_conda_packages
    else
        log_success "All packages verified"
    fi
}

# Setup compiler for macOS
setup_macos_compiler() {
    log_info "Setting up compiler for macOS..."
    
    # Check for Xcode Command Line Tools
    if ! xcode-select -p &> /dev/null; then
        log_warn "Xcode Command Line Tools not found."
        log_info "Installing Xcode Command Line Tools (this may take a while)..."
        xcode-select --install || {
            log_error "Failed to install Xcode Command Line Tools. Please install manually."
            exit 1
        }
        log_warn "Please complete the Xcode Command Line Tools installation, then re-run this script."
        exit 1
    fi
    
    # Find system compilers (check standard locations, not PATH which may have conda)
    # On macOS, system compilers are always in /usr/bin or CommandLineTools
    SYSTEM_CLANGXX=""
    if [[ -f "/usr/bin/clang++" ]]; then
        SYSTEM_CLANGXX="/usr/bin/clang++"
    elif [[ -f "/Library/Developer/CommandLineTools/usr/bin/clang++" ]]; then
        SYSTEM_CLANGXX="/Library/Developer/CommandLineTools/usr/bin/clang++"
    else
        log_error "System clang++ not found in standard locations"
        exit 1
    fi
    
    SYSTEM_CLANG=""
    if [[ -f "/usr/bin/clang" ]]; then
        SYSTEM_CLANG="/usr/bin/clang"
    elif [[ -f "/Library/Developer/CommandLineTools/usr/bin/clang" ]]; then
        SYSTEM_CLANG="/Library/Developer/CommandLineTools/usr/bin/clang"
    else
        log_error "System clang not found in standard locations"
        exit 1
    fi
    
    # Verify system compilers work
    if ! "${SYSTEM_CLANGXX}" --version &> /dev/null; then
        log_error "System clang++ not working. Please install Xcode Command Line Tools."
        exit 1
    fi
    
    CLANG_VERSION=$("${SYSTEM_CLANGXX}" --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1 | cut -d. -f1)
    if [[ "${CLANG_VERSION}" -lt 14 ]]; then
        log_warn "clang++ version ${CLANG_VERSION} is less than 14. This may cause issues."
    else
        log_success "System clang++ found: $("${SYSTEM_CLANGXX}" --version | head -n1)"
    fi
    
    # Remove conda compilers if present (they cause linker issues on macOS)
    if conda list | grep -q "cxx-compiler\|c-compiler"; then
        log_warn "Conda compilers detected. Removing to avoid linker issues..."
        conda remove --force cxx-compiler c-compiler -y 2>/dev/null || true
    fi
    
    # Remove conda clang/clang++ if present (they interfere with system compiler)
    if [[ -f "${CONDA_PREFIX}/bin/clang" ]] || [[ -f "${CONDA_PREFIX}/bin/clang++" ]]; then
        log_warn "Conda clang/clang++ detected. Removing to avoid linker issues..."
        conda remove --force clang clangxx -y 2>/dev/null || true
        # Also try removing via package name variations
        conda remove --force clang_osx-arm64 clangxx_osx-arm64 -y 2>/dev/null || true
        conda remove --force clang_osx-64 clangxx_osx-64 -y 2>/dev/null || true
    fi
    
    export CXX="${SYSTEM_CLANGXX}"
    export CC="${SYSTEM_CLANG}"
    export CMAKE_C_COMPILER="${SYSTEM_CLANG}"
    export CMAKE_CXX_COMPILER="${SYSTEM_CLANGXX}"
    log_info "Using system compiler: CXX=${CXX}, CC=${CC}"
}

# Setup compiler for Linux
setup_linux_compiler() {
    log_info "Setting up compiler for Linux..."
    
    if [[ "${USE_CONDA_COMPILERS}" == "true" ]]; then
        log_info "Using conda compilers..."
        # Verify compilers are installed
        if [[ ! -f "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
            log_warn "Conda compilers not found, installing..."
            conda install -c conda-forge cxx-compiler c-compiler -y
        fi
        # Use conda compilers
        export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
        export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
        export CMAKE_C_COMPILER="${CC}"
        export CMAKE_CXX_COMPILER="${CXX}"
        log_success "Conda compilers configured"
    else
        # Check for system compiler
        if command -v g++ &> /dev/null; then
            GCC_VERSION=$(g++ --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1 | cut -d. -f1)
            if [[ "${GCC_VERSION}" -lt 10 ]]; then
                log_warn "g++ version ${GCC_VERSION} is less than 10. Consider using conda compilers."
                log_info "Set USE_CONDA_COMPILERS=true to use conda compilers instead."
            else
                log_success "g++ found: $(g++ --version | head -n1)"
            fi
            SYSTEM_GXX=$(which g++)
            SYSTEM_GCC=$(which gcc)
            export CXX="${SYSTEM_GXX}"
            export CC="${SYSTEM_GCC}"
            export CMAKE_C_COMPILER="${CC}"
            export CMAKE_CXX_COMPILER="${CXX}"
        else
            log_warn "System g++ not found. Installing conda compilers..."
            conda install -c conda-forge cxx-compiler c-compiler -y
            export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
            export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
            export CMAKE_C_COMPILER="${CC}"
            export CMAKE_CXX_COMPILER="${CXX}"
            log_success "Conda compilers installed and configured"
        fi
    fi
    
    log_info "Using compiler: CXX=${CXX}, CC=${CC}"
}

# Setup compiler for Windows
setup_windows_compiler() {
    log_info "Setting up compiler for Windows..."
    
    if [[ "${USE_CONDA_COMPILERS}" == "true" ]]; then
        log_info "Using conda compilers..."
        # Verify compilers are installed
        if ! conda list | grep -q "cxx-compiler"; then
            log_warn "Conda compilers not found, installing..."
            conda install -c conda-forge cxx-compiler c-compiler -y
        fi
        log_success "Conda compilers configured"
    else
        # Check for MSVC (Visual Studio)
        if command -v cl &> /dev/null; then
            log_success "MSVC compiler found"
        else
            log_warn "MSVC compiler not found in PATH."
            log_info "Options:"
            log_info "  1. Install Visual Studio Build Tools and use 'x64 Native Tools Command Prompt'"
            log_info "  2. Set USE_CONDA_COMPILERS=true to use conda compilers"
            read -p "Install conda compilers now? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                log_info "Installing conda compilers..."
                conda install -c conda-forge cxx-compiler c-compiler -y
                log_success "Conda compilers installed"
            else
                log_warn "Proceeding without compiler. You may need to set up MSVC manually."
            fi
        fi
    fi
}

# Verify environment
verify_environment() {
    log_info "Verifying environment setup..."
    
    local errors=0
    
    # Check Python
    if python --version &> /dev/null; then
        PYTHON_VER=$(python --version | cut -d' ' -f2 | cut -d. -f1,2)
        log_success "Python: $(python --version)"
    else
        log_error "Python not found"
        ((errors++))
    fi
    
    # Check CMake
    if cmake --version &> /dev/null; then
        CMAKE_VER=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
        CMAKE_MAJOR=$(echo "${CMAKE_VER}" | cut -d. -f1)
        CMAKE_MINOR=$(echo "${CMAKE_VER}" | cut -d. -f2)
        if [[ "${CMAKE_MAJOR}" -gt 3 ]] || [[ "${CMAKE_MAJOR}" -eq 3 && "${CMAKE_MINOR}" -ge 20 ]]; then
            log_success "CMake: $(cmake --version | head -n1)"
        else
            log_error "CMake version ${CMAKE_VER} is less than 3.20"
            ((errors++))
        fi
    else
        log_error "CMake not found"
        ((errors++))
    fi
    
    # Check Ninja
    if ninja --version &> /dev/null; then
        log_success "Ninja: $(ninja --version)"
    else
        log_warn "Ninja not found"
    fi
    
    # Check Git
    if git --version &> /dev/null; then
        log_success "Git: $(git --version)"
    else
        log_error "Git not found"
        ((errors++))
    fi
    
    # Check compiler (platform-specific)
    case "${PLATFORM}" in
        macos)
            if clang++ --version &> /dev/null; then
                log_success "C++ Compiler: $(clang++ --version | head -n1)"
            else
                log_error "clang++ not found"
                ((errors++))
            fi
            ;;
        linux)
            if command -v "${CXX:-g++}" &> /dev/null; then
                log_success "C++ Compiler: $(${CXX:-g++} --version | head -n1)"
            else
                log_error "C++ compiler not found"
                ((errors++))
            fi
            ;;
        windows)
            if command -v cl &> /dev/null || conda list | grep -q "cxx-compiler"; then
                log_success "C++ Compiler: Available"
            else
                log_warn "C++ compiler may not be properly configured"
            fi
            ;;
    esac
    
    # Check LAPACKE if installed
    if [[ "${INSTALL_LAPACKE}" == "true" ]]; then
        if [[ -f "${CONDA_PREFIX}/include/lapacke.h" ]] || conda list | grep -q "lapacke"; then
            log_success "LAPACKE: Installed"
        else
            log_warn "LAPACKE: Not found (may still work without it)"
        fi
    fi
    
    if [[ ${errors} -gt 0 ]]; then
        log_error "Environment verification failed with ${errors} error(s)"
        return 1
    else
        log_success "Environment verification passed"
        return 0
    fi
}

# Get paths to tools in conda environment
get_conda_tool_paths() {
    # Ensure we're using tools from the conda environment
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        # Python and CMake should always be in conda environment
        export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
        export CMAKE_BIN="${CONDA_PREFIX}/bin/cmake"
        
        # Ninja - prefer conda, fallback to system
        if [[ -f "${CONDA_PREFIX}/bin/ninja" ]]; then
            export NINJA_BIN="${CONDA_PREFIX}/bin/ninja"
        elif command -v ninja &> /dev/null; then
            export NINJA_BIN="$(command -v ninja)"
            log_warn "Using system ninja instead of conda version"
        else
            export NINJA_BIN="ninja"
        fi
        
        # Git - prefer conda, fallback to system
        if [[ -f "${CONDA_PREFIX}/bin/git" ]]; then
            export GIT_BIN="${CONDA_PREFIX}/bin/git"
        elif command -v git &> /dev/null; then
            export GIT_BIN="$(command -v git)"
            log_warn "Using system git instead of conda version"
        else
            export GIT_BIN="git"
        fi
        
        # Add conda bin to PATH to ensure tools are found
        export PATH="${CONDA_PREFIX}/bin:${PATH}"
        
        log_info "Using conda environment tools from: ${CONDA_PREFIX}/bin"
    else
        log_error "CONDA_PREFIX not set. Please activate the conda environment first."
        exit 1
    fi
}

# Check for a working liblapacke installation
has_lapacke() {
    local check_dir
    # Temporarily disable unbound variable checking for mktemp
    set +u
    check_dir=$(mktemp -d 2>/dev/null || mktemp -d -t 'check_lapacke' 2>/dev/null || echo "")
    set -u
    
    if [[ -z "${check_dir}" ]] || [[ ! -d "${check_dir}" ]]; then
        log_warn "Failed to create temporary directory for LAPACKE check. Assuming not present."
        return 1
    fi
    
    trap 'rm -rf -- "${check_dir:-}"' RETURN

    cat > "${check_dir}/check.cpp" <<EOF
#include <lapacke.h>
int main() {
    // We don't need to run it, just check if it links.
    // Calling a function prevents the linker from optimizing the library away.
    (void)LAPACKE_dgesv;
    return 0;
}
EOF

    local CXX_COMPILER="${CXX:-g++}"
    if ! command -v "${CXX_COMPILER}" &> /dev/null; then
        CXX_COMPILER="c++"
        if ! command -v "${CXX_COMPILER}" &> /dev/null; then
            log_warn "No C++ compiler found to check for LAPACKE. Assuming not present."
            return 1
        fi
    fi

    local CXX_FLAGS=""
    local LDFLAGS=""
    # If we are in a conda environment, add its paths
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        CXX_FLAGS="-I${CONDA_PREFIX}/include"
        LDFLAGS="-L${CONDA_PREFIX}/lib"
    fi

    # Attempt to compile and link against liblapacke.
    if "${CXX_COMPILER}" "${check_dir}/check.cpp" -o "${check_dir}/check.out" ${CXX_FLAGS} ${LDFLAGS} -llapacke &> /dev/null; then
        return 0 # Success
    else
        return 1 # Failure
    fi
}

# Install QUGaR
install_qugar() {
    log_info "Starting QUGaR installation..."
    
    # Get paths to conda tools
    get_conda_tool_paths
    
    # Set include paths for LAPACKE if in conda environment
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "${CONDA_PREFIX}/include/lapacke.h" ]]; then
        export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH:-}"
        export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
        export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    fi
    
    # Verify tools are available
    if ! command -v "${CMAKE_BIN}" &> /dev/null; then
        log_error "cmake not found in conda environment. Please install CMake >= 3.20."
        exit 1
    fi
    
    if ! command -v "${PYTHON_BIN}" &> /dev/null; then
        log_error "python not found in conda environment."
        exit 1
    fi
    
    if ! command -v "${GIT_BIN}" &> /dev/null; then
        log_error "git not found. Please install git."
        exit 1
    fi
    
    # Determine generator
    GENERATOR="Ninja"
    if ! command -v "${NINJA_BIN}" &> /dev/null; then
        log_warn "Ninja not found. Falling back to Unix Makefiles generator."
        GENERATOR="Unix Makefiles"
    fi
    
    log_info "Using generator: ${GENERATOR}"
    
    # Setup paths
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DEPS_DIR="${REPO_ROOT}/.deps"
    QUGAR_DIR="${DEPS_DIR}/qugar"
    BUILD_DIR="${QUGAR_DIR}/build"
    INSTALL_DIR="${QUGAR_DIR}/install"
    
    PIP_CMD=("${PYTHON_BIN}" "-m" "pip")
    JOBS="${JOBS:-$(command -v sysctl &> /dev/null && sysctl -n hw.ncpu || nproc || echo 4)}"
    
    # Always clean .deps folder for fresh start
    if [[ -d "${DEPS_DIR}" ]]; then
        log_info "Cleaning .deps folder for fresh clone..."
        rm -rf "${DEPS_DIR}"
    fi
    
    log_info "Preparing dependencies directory at ${DEPS_DIR}"
    mkdir -p "${DEPS_DIR}"
    
    # Set environment variables early so CMake and compilers can find LAPACKE
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "${CONDA_PREFIX}/include/lapacke.h" ]]; then
        export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH:-}"
        export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
        export CPPFLAGS="-I${CONDA_PREFIX}/include ${CPPFLAGS:-}"
        export LDFLAGS="-L${CONDA_PREFIX}/lib ${LDFLAGS:-}"
        log_info "Set environment variables for compiler (CPLUS_INCLUDE_PATH, C_INCLUDE_PATH, CPPFLAGS, LDFLAGS)"
    fi
    
    # Always do a fresh clone of QUGaR repository
    if [[ -d "${QUGAR_DIR}" ]]; then
        log_info "Removing existing qugar directory..."
        rm -rf "${QUGAR_DIR}"
    fi
    log_info "Cloning QUGaR repository..."
    "${GIT_BIN}" clone https://github.com/pantolin/qugar.git "${QUGAR_DIR}"
    
    # Create build directory after cloning
    mkdir -p "${BUILD_DIR}"
    
    # Apply tpms_lib.hpp fix
    TPMS_HEADER="${QUGAR_DIR}/cpp/include/qugar/tpms_lib.hpp"
    if [ -f "${TPMS_HEADER}" ]; then
        log_info "Ensuring tpms_lib.hpp has dependent-type qualifiers..."
        "${PYTHON_BIN}" - "$TPMS_HEADER" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
original = path.read_text()

patched = original.replace(
    "template<typename T> using Gradient = qugar::impl::ImplicitFunc<dim>::template Gradient<T>;",
    "template<typename T> using Gradient = typename qugar::impl::ImplicitFunc<dim>::template Gradient<T>;"
)
patched = patched.replace(
    "template<typename T> using Hessian = qugar::impl::ImplicitFunc<dim>::template Hessian<T>;",
    "template<typename T> using Hessian = typename qugar::impl::ImplicitFunc<dim>::template Hessian<T>;"
)

if patched != original:
    path.write_text(patched)
    print("[INFO]   Applied typename fix to tpms aliases.")
else:
    print("[INFO]   tpms aliases already patched.")
PY
    fi
    
    # Check for liblapacke and prepare CMake arguments
    CMAKE_EXTRA_ARGS=()
    if has_lapacke; then
        log_info "Found liblapacke. Configuring QUGaR with LAPACKE support."
        CMAKE_EXTRA_ARGS+=("-DQUGAR_WITH_LAPACKE=ON")
        
        # Add LAPACKE include and library paths from conda environment
        if [[ -n "${CONDA_PREFIX:-}" ]]; then
            if [[ -f "${CONDA_PREFIX}/include/lapacke.h" ]]; then
                # Add conda prefix to CMAKE_PREFIX_PATH to help CMake find LAPACKE
                export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
                
                # Explicitly set LAPACKE paths for CMake
                CMAKE_EXTRA_ARGS+=("-DLAPACKE_INCLUDE_DIR=${CONDA_PREFIX}/include")
                
                # Find the actual library files (handle different extensions)
                LAPACKE_LIB=""
                if [[ -f "${CONDA_PREFIX}/lib/liblapacke.dylib" ]]; then
                    LAPACKE_LIB="${CONDA_PREFIX}/lib/liblapacke.dylib"
                elif [[ -f "${CONDA_PREFIX}/lib/liblapacke.a" ]]; then
                    LAPACKE_LIB="${CONDA_PREFIX}/lib/liblapacke.a"
                elif [[ -f "${CONDA_PREFIX}/lib/liblapacke.so" ]]; then
                    LAPACKE_LIB="${CONDA_PREFIX}/lib/liblapacke.so"
                fi
                
                if [[ -n "${LAPACKE_LIB}" ]]; then
                    # Build library list - LAPACKE depends on LAPACK and BLAS
                    LAPACKE_LIBS="${LAPACKE_LIB}"
                    if [[ -f "${CONDA_PREFIX}/lib/libopenblas.dylib" ]] || [[ -f "${CONDA_PREFIX}/lib/libopenblas.a" ]] || [[ -f "${CONDA_PREFIX}/lib/libopenblas.so" ]]; then
                        # OpenBLAS provides both BLAS and LAPACK
                        if [[ -f "${CONDA_PREFIX}/lib/libopenblas.dylib" ]]; then
                            LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/libopenblas.dylib"
                        elif [[ -f "${CONDA_PREFIX}/lib/libopenblas.a" ]]; then
                            LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/libopenblas.a"
                        else
                            LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/libopenblas.so"
                        fi
                    else
                        # Try separate LAPACK and BLAS libraries
                        if [[ -f "${CONDA_PREFIX}/lib/liblapack.dylib" ]] || [[ -f "${CONDA_PREFIX}/lib/liblapack.a" ]]; then
                            if [[ -f "${CONDA_PREFIX}/lib/liblapack.dylib" ]]; then
                                LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/liblapack.dylib"
                            else
                                LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/liblapack.a"
                            fi
                        fi
                        if [[ -f "${CONDA_PREFIX}/lib/libblas.dylib" ]] || [[ -f "${CONDA_PREFIX}/lib/libblas.a" ]]; then
                            if [[ -f "${CONDA_PREFIX}/lib/libblas.dylib" ]]; then
                                LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/libblas.dylib"
                            else
                                LAPACKE_LIBS="${LAPACKE_LIBS};${CONDA_PREFIX}/lib/libblas.a"
                            fi
                        fi
                    fi
                    
                    CMAKE_EXTRA_ARGS+=("-DLAPACKE_LIBRARIES=${LAPACKE_LIBS}")
                    log_info "Using LAPACKE from conda environment: ${CONDA_PREFIX}"
                fi
            fi
        fi
    else
        log_info "liblapacke not found or not usable. Building QUGaR without LAPACKE support."
    fi
    
    log_info "Configuring C++ core build..."
    # Build cmake command with conditional extra args and explicit compiler paths
    CMAKE_CMD=(
        "${CMAKE_BIN}"
        -G "${GENERATOR}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    )
    
    # Add explicit compiler paths if set
    if [[ -n "${CMAKE_C_COMPILER:-}" ]]; then
        CMAKE_CMD+=("-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
    fi
    if [[ -n "${CMAKE_CXX_COMPILER:-}" ]]; then
        CMAKE_CMD+=("-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    fi
    
    # Add conda include directory to compiler flags
    # Use CMAKE_CXX_FLAGS_INIT which is processed before cache
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "${CONDA_PREFIX}/include/lapacke.h" ]]; then
        CMAKE_CMD+=("-DCMAKE_CXX_FLAGS_INIT=-I${CONDA_PREFIX}/include")
        CMAKE_CMD+=("-DCMAKE_C_FLAGS_INIT=-I${CONDA_PREFIX}/include")
        CMAKE_CMD+=("-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}")
        CMAKE_CMD+=("-DCMAKE_INCLUDE_PATH=${CONDA_PREFIX}/include")
        log_info "Adding conda paths to CMake: -I${CONDA_PREFIX}/include"
    fi
    
    # Add extra args only if array is not empty
    if [ ${#CMAKE_EXTRA_ARGS[@]} -gt 0 ]; then
        CMAKE_CMD+=("${CMAKE_EXTRA_ARGS[@]}")
    fi
    
    CMAKE_CMD+=(
        -B "${BUILD_DIR}"
        -S "${QUGAR_DIR}/cpp"
    )
    
    "${CMAKE_CMD[@]}"
    
    log_info "Building C++ core with ${JOBS} parallel jobs..."
    "${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${JOBS}"
    
    log_info "Installing C++ artifacts to ${INSTALL_DIR}"
    "${CMAKE_BIN}" --install "${BUILD_DIR}"
    
    export CMAKE_PREFIX_PATH="${INSTALL_DIR}:${CMAKE_PREFIX_PATH:-}"
    
    PYTHON_DIR="${QUGAR_DIR}/python"
    
    if [ ! -f "${PYTHON_DIR}/build-requirements.txt" ]; then
        log_error "Missing build requirements file at ${PYTHON_DIR}/build-requirements.txt"
        exit 1
    fi
    
    log_info "Installing Python build requirements..."
    "${PIP_CMD[@]}" -v install -r "${PYTHON_DIR}/build-requirements.txt"
    
    log_info "Installing QUGaR Python interface..."
    "${PIP_CMD[@]}" -v install --no-build-isolation "${PYTHON_DIR}" -U
    
    log_success "QUGaR installation complete."
    
    log_info "Cleaning up build dependencies..."
    cd "${REPO_ROOT}"
    rm -rf "${DEPS_DIR}"
    
    log_success "QUGaR installation and cleanup complete."
}

# Main execution
main() {
    echo "=========================================="
    echo "  QUGaR Complete Setup and Installation"
    echo "=========================================="
    echo
    
    detect_platform
    check_prerequisites
    
    # Setup environment if not skipped
    if [[ "${SKIP_ENV_SETUP}" != "true" ]]; then
        if ! check_existing_env; then
            create_conda_env
        fi
        
        # Verify all packages are installed
        verify_packages
         
        # Platform-specific compiler setup
        case "${PLATFORM}" in
            macos)
                setup_macos_compiler
                ;;
            linux)
                setup_linux_compiler
                ;;
            windows)
                setup_windows_compiler
                ;;
        esac
        
        if ! verify_environment; then
            log_error "Environment setup completed with errors. Please review the output above."
            exit 1
        fi
    else
        # Just activate existing environment
        log_info "Skipping environment setup. Activating existing environment..."
        activate_conda_env
        
        # Still need to setup compiler
        case "${PLATFORM}" in
            macos)
                setup_macos_compiler
                ;;
            linux)
                setup_linux_compiler
                ;;
            windows)
                setup_windows_compiler
                ;;
        esac
    fi
    
    # Install QUGaR
    install_qugar
    
    log_success "=========================================="
    log_success "  QUGaR setup and installation complete!"
    log_success "=========================================="
    echo
    log_info "To use QUGaR:"
    echo "  1. Activate the environment:"
    echo "     conda activate ${ENV_NAME}"
    echo
    echo "  2. Launch Python and import qugar:"
    echo "     python"
    echo "     >>> import qugar"
    echo
    echo "  Or run Python directly:"
    echo "     python -c \"import qugar; print(qugar.__version__)\""
}

# Run main function
main "$@"
