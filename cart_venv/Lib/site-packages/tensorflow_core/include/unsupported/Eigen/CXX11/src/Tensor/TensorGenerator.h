// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_GENERATOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_GENERATOR_H

namespace Eigen {

/** \class TensorGeneratorOp
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor generator class.
  *
  *
  */
namespace internal {
template<typename Generator, typename XprType>
struct traits<TensorGeneratorOp<Generator, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
  typedef typename XprTraits::PointerType PointerType;
};

template<typename Generator, typename XprType>
struct eval<TensorGeneratorOp<Generator, XprType>, Eigen::Dense>
{
  typedef const TensorGeneratorOp<Generator, XprType>& type;
};

template<typename Generator, typename XprType>
struct nested<TensorGeneratorOp<Generator, XprType>, 1, typename eval<TensorGeneratorOp<Generator, XprType> >::type>
{
  typedef TensorGeneratorOp<Generator, XprType> type;
};

}  // end namespace internal



template<typename Generator, typename XprType>
class TensorGeneratorOp : public TensorBase<TensorGeneratorOp<Generator, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorGeneratorOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorGeneratorOp(const XprType& expr, const Generator& generator)
      : m_xpr(expr), m_generator(generator) {}

    EIGEN_DEVICE_FUNC
    const Generator& generator() const { return m_generator; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
    const Generator m_generator;
};


// Eval as rvalue
template<typename Generator, typename ArgType, typename Device>
struct TensorEvaluator<const TensorGeneratorOp<Generator, ArgType>, Device>
{
  typedef TensorGeneratorOp<Generator, ArgType> XprType;
  typedef typename XprType::Index Index;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;
  static const int NumDims = internal::array_size<Dimensions>::value;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef StorageMemory<CoeffReturnType, Device> Storage;
  typedef typename Storage::Type EvaluatorPointerType;
  enum {
    IsAligned         = false,
    PacketAccess      = (PacketType<CoeffReturnType, Device>::size > 1),
    BlockAccess       = true,
    PreferBlockAccess = true,
    Layout            = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess       = false,  // to be implemented
    RawAccess         = false
  };

  typedef internal::TensorIntDivisor<Index> IndexDivisor;

  typedef internal::TensorBlock<CoeffReturnType, Index, NumDims, Layout>
      TensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      :  m_device(device), m_generator(op.generator())
  {
    TensorEvaluator<ArgType, Device> argImpl(op.expression(), device);
    m_dimensions = argImpl.dimensions();

    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_strides[0] = 1;
      EIGEN_UNROLL_LOOP
      for (int i = 1; i < NumDims; ++i) {
        m_strides[i] = m_strides[i - 1] * m_dimensions[i - 1];
        if (m_strides[i] != 0) m_fast_strides[i] = IndexDivisor(m_strides[i]);
      }
    } else {
      m_strides[NumDims - 1] = 1;
      EIGEN_UNROLL_LOOP
      for (int i = NumDims - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i + 1] * m_dimensions[i + 1];
        if (m_strides[i] != 0) m_fast_strides[i] = IndexDivisor(m_strides[i]);
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType /*data*/) {
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    array<Index, NumDims> coords;
    extract_coordinates(index, coords);
    return m_generator(coords);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = PacketType<CoeffReturnType, Device>::size;
    EIGEN_STATIC_ASSERT((packetSize > 1), YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
    Eigen::Index block_total_size_max = numext::maxi<Eigen::Index>(
        1, m_device.firstLevelCacheSize() / sizeof(Scalar));
    resources->push_back(internal::TensorOpResourceRequirements(
        internal::kSkewedInnerDims, block_total_size_max));
  }

  struct BlockIteratorState {
    Index stride;
    Index span;
    Index size;
    Index count;
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(
      TensorBlock* output_block) const {
    if (NumDims <= 0) return;

    static const bool is_col_major =
        static_cast<int>(Layout) == static_cast<int>(ColMajor);

    // Compute spatial coordinates for the first block element.
    array<Index, NumDims> coords;
    extract_coordinates(output_block->first_coeff_index(), coords);
    array<Index, NumDims> initial_coords = coords;

    CoeffReturnType* data = output_block->data();
    Index offset = 0;

    // Initialize output block iterator state. Dimension in this array are
    // always in inner_most -> outer_most order (col major layout).
    array<BlockIteratorState, NumDims> it;
    for (Index i = 0; i < NumDims; ++i) {
      const Index dim = is_col_major ? i : NumDims - 1 - i;
      it[i].size = output_block->block_sizes()[dim];
      it[i].stride = output_block->block_strides()[dim];
      it[i].span = it[i].stride * (it[i].size - 1);
      it[i].count = 0;
    }
    eigen_assert(it[0].stride == 1);

    while (it[NumDims - 1].count < it[NumDims - 1].size) {
      // Generate data for the inner-most dimension.
      for (Index i = 0; i < it[0].size; ++i) {
        *(data + offset + i) = m_generator(coords);
        coords[is_col_major ? 0 : NumDims - 1]++;
      }
      coords[is_col_major ? 0 : NumDims - 1] =
          initial_coords[is_col_major ? 0 : NumDims - 1];

      // For the 1d tensor we need to generate only one inner-most dimension.
      if (NumDims == 1) break;

      // Update offset.
      for (Index i = 1; i < NumDims; ++i) {
        if (++it[i].count < it[i].size) {
          offset += it[i].stride;
          coords[is_col_major ? i : NumDims - 1 - i]++;
          break;
        }
        if (i != NumDims - 1) it[i].count = 0;
        coords[is_col_major ? i : NumDims - 1 - i] =
            initial_coords[is_col_major ? i : NumDims - 1 - i];
        offset -= it[i].span;
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool) const {
    // TODO(rmlarsen): This is just a placeholder. Define interface to make
    // generators return their cost.
    return TensorOpCost(0, 0, TensorOpCost::AddCost<Scalar>() +
                                  TensorOpCost::MulCost<Scalar>());
  }

  EIGEN_DEVICE_FUNC EvaluatorPointerType  data() const { return NULL; }

#ifdef EIGEN_USE_SYCL
  // binding placeholder accessors to a command group handler for SYCL
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void bind(cl::sycl::handler&) const {}
#endif

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void extract_coordinates(Index index, array<Index, NumDims>& coords) const {
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx = index / m_fast_strides[i];
        index -= idx * m_strides[i];
        coords[i] = idx;
      }
      coords[0] = index;
    } else {
      for (int i = 0; i < NumDims - 1; ++i) {
        const Index idx = index / m_fast_strides[i];
        index -= idx * m_strides[i];
        coords[i] = idx;
      }
      coords[NumDims-1] = index;
    }
  }

  const Device EIGEN_DEVICE_REF m_device;
  Dimensions m_dimensions;
  array<Index, NumDims> m_strides;
  array<IndexDivisor, NumDims> m_fast_strides;
  Generator m_generator;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_GENERATOR_H
