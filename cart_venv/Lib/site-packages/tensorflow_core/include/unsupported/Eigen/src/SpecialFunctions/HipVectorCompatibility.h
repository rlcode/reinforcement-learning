#ifndef HIP_VECTOR_COMPATIBILITY_H
#define HIP_VECTOR_COMPATIBILITY_H

namespace hip_impl {
  template <typename, typename, unsigned int> struct Scalar_accessor;
}   // end namespace hip_impl

namespace Eigen {
namespace internal {

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct lgamma_impl<hip_impl::Scalar_accessor<T, U, n>> : lgamma_impl<T> {};
#endif

template <typename T, typename U, unsigned int n>
struct digamma_impl_maybe_poly<hip_impl::Scalar_accessor<T, U, n>>
  : digamma_impl_maybe_poly<T> {};

template <typename T, typename U, unsigned int n>
struct digamma_impl<hip_impl::Scalar_accessor<T, U, n>> : digamma_impl<T> {};

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct erf_impl<hip_impl::Scalar_accessor<T, U, n>> : erf_impl<T> {};
#endif  // EIGEN_HAS_C99_MATH

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct erfc_impl<hip_impl::Scalar_accessor<T, U, n>> : erfc_impl<T> {};
#endif  // EIGEN_HAS_C99_MATH

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct ndtri_impl<hip_impl::Scalar_accessor<T, U, n>> : ndtri_impl<T> {};
#endif  // EIGEN_HAS_C99_MATH

template <typename T, typename U, unsigned int n, IgammaComputationMode mode>
struct igammac_cf_impl<hip_impl::Scalar_accessor<T, U, n>, mode>
  : igammac_cf_impl<T, mode> {};

template <typename T, typename U, unsigned int n, IgammaComputationMode mode>
struct igamma_series_impl<hip_impl::Scalar_accessor<T, U, n>, mode>
  : igamma_series_impl<T, mode> {};

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct igammac_impl<hip_impl::Scalar_accessor<T, U, n>> : igammac_impl<T> {};
#endif  // EIGEN_HAS_C99_MATH

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n, IgammaComputationMode mode>
struct igamma_generic_impl<hip_impl::Scalar_accessor<T, U, n>, mode>
  : igamma_generic_impl<T, mode> {};
#endif  // EIGEN_HAS_C99_MATH

template <typename T, typename U, unsigned int n>
struct igamma_impl<hip_impl::Scalar_accessor<T, U, n>> : igamma_impl<T> {};

template <typename T, typename U, unsigned int n>
struct igamma_der_a_retval<hip_impl::Scalar_accessor<T, U, n>>
  : igamma_der_a_retval<T> {};

template <typename T, typename U, unsigned int n>
struct igamma_der_a_impl<hip_impl::Scalar_accessor<T, U, n>>
  : igamma_der_a_impl<T> {};

template <typename T, typename U, unsigned int n>
struct gamma_sample_der_alpha_retval<hip_impl::Scalar_accessor<T, U, n>>
  : gamma_sample_der_alpha_retval<T> {};

template <typename T, typename U, unsigned int n>
struct gamma_sample_der_alpha_impl<hip_impl::Scalar_accessor<T, U, n>>
  : gamma_sample_der_alpha_impl<T> {};

template <typename T, typename U, unsigned int n>
struct zeta_impl_series<hip_impl::Scalar_accessor<T, U, n>>
  : zeta_impl_series<T> {};

template <typename T, typename U, unsigned int n>
struct zeta_impl<hip_impl::Scalar_accessor<T, U, n>> : zeta_impl<T> {};

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct polygamma_impl<hip_impl::Scalar_accessor<T, U, n>>
  : polygamma_impl<T> {};
#endif  // EIGEN_HAS_C99_MATH

#if EIGEN_HAS_C99_MATH
template <typename T, typename U, unsigned int n>
struct betainc_impl<hip_impl::Scalar_accessor<T, U, n>> : betainc_impl<T> {};

template <typename T, typename U, unsigned int n>
struct incbeta_cfe<hip_impl::Scalar_accessor<T, U, n>> : incbeta_cfe<T> {};

template <typename T, typename U, unsigned int n>
struct betainc_helper<hip_impl::Scalar_accessor<T, U, n>>
  : betainc_helper<T> {};
#else
template <typename T, typename U, unsigned int n>
struct betainc_impl<hip_impl::Scalar_accessor<T, U, n>> : betainc_impl<T> {};
#endif  // EIGEN_HAS_C99_MATH

template <typename T, typename U, unsigned int n>
struct i0e_impl<hip_impl::Scalar_accessor<T, U, n>> : i0e_impl<T> {};

template <typename T, typename U, unsigned int n>
struct i1e_impl<hip_impl::Scalar_accessor<T, U, n>> : i1e_impl<T> {};

}  // end namespace internal
}  // end namespace Eigen

#endif  // HIP_VECTOR_COMPATIBILITY_H
