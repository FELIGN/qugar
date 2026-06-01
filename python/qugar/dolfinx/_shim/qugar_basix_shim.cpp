// A tiny, generic C++ shim that lets a JIT-compiled C kernel tabulate basix
// finite elements on the fly.  It is compiled once, cached on disk, and linked
// into every qugar custom kernel.
//
// Design invariants:
//   1. basix (C++) is callable from a plain C ABI (extern "C").
//   2. Each element is created ONCE and cached; tabulate never re-constructs it.
//   3. Scalar-type awareness: basix elements are templated on the *real* type
//      (float/double). Tabulation is always real-valued; the kernel's assembly
//      type (which may be complex) is a separate concern handled on the C side.
//   4. No C++ exception ever crosses the C boundary (all guarded -> error codes).
//
// The shim is intentionally element-agnostic: it knows nothing about FFCx table
// layouts. The basix->FFCx (derivative/component) repack lives in generated C.

#include <basix/finite-element.h>

#include <array>
#include <cstddef>
#include <deque>
#include <exception>
#include <map>
#include <span>
#include <utility>
#include <vector>

using namespace basix;

namespace {
// One registry per real type. Static => created once, lives for the process,
// freed via qugar_shim_reset(). Handles are indices into these vectors.
template<typename T> std::vector<FiniteElement<T>> &registry()
{
  static std::vector<FiniteElement<T>> r;
  return r;
}

// Dedup index: (family,cell,degree,lvariant,dvariant,discontinuous) -> handle.
// The generated kernel calls register on every cell; dedup guarantees the
// element is created exactly ONCE and later calls are a cheap map lookup.
using ElementKey = std::array<int, 6>;

template<typename T> std::map<ElementKey, int> &index()
{
  static std::map<ElementKey, int> idx;
  return idx;
}

template<typename T> int register_impl(int family, int cell, int degree, int lvariant, int dvariant, int discontinuous)
{
  const ElementKey key{ family, cell, degree, lvariant, dvariant, discontinuous };
  if (auto it = index<T>().find(key); it != index<T>().end())
    return it->second;// already created -> return cached handle

  try {
    FiniteElement<T> el = create_element<T>(static_cast<element::family>(family),
      static_cast<cell::type>(cell),
      degree,
      static_cast<element::lagrange_variant>(lvariant),
      static_cast<element::dpc_variant>(dvariant),
      static_cast<bool>(discontinuous));
    registry<T>().push_back(std::move(el));
    const int handle = static_cast<int>(registry<T>().size()) - 1;
    index<T>()[key] = handle;
    return handle;
  } catch (const std::exception &) {
    return -1;// basix rejected the parameters
  } catch (...) {
    return -2;// unknown failure
  }
}

template<typename T> int shape_impl(int handle, int nd, int npts, long *out4)
{
  try {
    auto &r = registry<T>();
    if (handle < 0 || handle >= static_cast<int>(r.size()))
      return -3;
    std::array<std::size_t, 4> s =
      r[handle].tabulate_shape(static_cast<std::size_t>(nd), static_cast<std::size_t>(npts));
    for (int i = 0; i < 4; ++i)
      out4[i] = static_cast<long>(s[i]);
    return 0;
  } catch (...) {
    return -1;
  }
}

template<typename T>
int tabulate_impl(int handle, int nd, const T *x, int npts, int gdim, T *basis, long basis_capacity)
{
  try {
    auto &r = registry<T>();
    if (handle < 0 || handle >= static_cast<int>(r.size()))
      return -3;
    const FiniteElement<T> &el = r[handle];
    std::array<std::size_t, 4> s = el.tabulate_shape(static_cast<std::size_t>(nd), static_cast<std::size_t>(npts));
    const std::size_t need = s[0] * s[1] * s[2] * s[3];
    if (static_cast<long>(need) > basis_capacity)
      return -4;// caller's buffer too small
    el.tabulate(nd,
      std::span<const T>(x, static_cast<std::size_t>(npts) * gdim),
      { static_cast<std::size_t>(npts), static_cast<std::size_t>(gdim) },
      std::span<T>(basis, need));
    return 0;
  } catch (const std::exception &) {
    return -1;
  } catch (...) {
    return -2;
  }
}
// Thread-local scratch buffers used by the generated kernel for basix
// blocks and per-table repack buffers. Live for the thread's lifetime
// and grow on demand, so the kernel doesn't have to put large VLAs on
// the stack -- which on macOS overflows the (smaller) worker-thread
// stack for higher-degree 3D forms.
//
// There is ONE buffer per ``slot``: a kernel that assembles several
// quadratures (e.g. a combined volume ``dx`` + unfitted-boundary ``dsu``
// form) tabulates every quadrature's FE tables up front and reads them
// back in the quadrature loops that follow, so all of them must stay
// live simultaneously. A single shared buffer would alias them, and a
// later (larger) request could reallocate it out from under an earlier
// quadrature's still-held pointer. A ``std::deque`` never relocates its
// existing elements when a new slot is appended, so pointers handed out
// for earlier slots remain valid; each slot's own vector is only resized
// (at most once per kernel call) before its pointer is used.
template<typename T> T *get_scratch_impl(int slot, long n)
{
  thread_local std::deque<std::vector<T>> pool;
  if (slot < 0)
    return nullptr;
  while (static_cast<std::size_t>(slot) >= pool.size())
    pool.emplace_back();
  std::vector<T> &s = pool[static_cast<std::size_t>(slot)];
  try {
    if (s.size() < static_cast<std::size_t>(n))
      s.resize(static_cast<std::size_t>(n));
    return s.data();
  } catch (...) {
    return nullptr;
  }
}
}// namespace

extern "C" {
// --- float64 (double) entry points ---
int qugar_register_element_f64(int family, int cell, int degree, int lvariant, int dvariant, int discontinuous)
{
  return register_impl<double>(family, cell, degree, lvariant, dvariant, discontinuous);
}
int qugar_tabulate_shape_f64(int handle, int nd, int npts, long *out4)
{
  return shape_impl<double>(handle, nd, npts, out4);
}
int qugar_tabulate_f64(int handle, int nd, const double *x, int npts, int gdim, double *basis, long basis_capacity)
{
  return tabulate_impl<double>(handle, nd, x, npts, gdim, basis, basis_capacity);
}

// --- float32 (float) entry points ---
int qugar_register_element_f32(int family, int cell, int degree, int lvariant, int dvariant, int discontinuous)
{
  return register_impl<float>(family, cell, degree, lvariant, dvariant, discontinuous);
}
int qugar_tabulate_shape_f32(int handle, int nd, int npts, long *out4)
{
  return shape_impl<float>(handle, nd, npts, out4);
}
int qugar_tabulate_f32(int handle, int nd, const float *x, int npts, int gdim, float *basis, long basis_capacity)
{
  return tabulate_impl<float>(handle, nd, x, npts, gdim, basis, basis_capacity);
}

// Thread-local scratch buffer accessors used by the generated kernel.
// ``slot`` selects a per-quadrature buffer (see get_scratch_impl).
double *qugar_get_scratch_f64(int slot, long n)
{
  return get_scratch_impl<double>(slot, n);
}
float *qugar_get_scratch_f32(int slot, long n)
{
  return get_scratch_impl<float>(slot, n);
}

// Introspection / teardown.
int qugar_registry_size_f64()
{
  return (int)registry<double>().size();
}
int qugar_registry_size_f32()
{
  return (int)registry<float>().size();
}
void qugar_shim_reset()
{
  registry<double>().clear();
  registry<float>().clear();
  index<double>().clear();
  index<float>().clear();
}
}
