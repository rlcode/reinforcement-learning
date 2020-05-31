// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Navdeep Jaitly <ndjaitly@google.com>
//                    Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REVERSE_H
#define EIGEN_CXX11_TENSOR_TENSOR_REVERSE_H
namespace Eigen {

/** \class TensorReverse
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reverse elements class.
  *
  */
namespace internal {
template<typename ReverseDimensions, typename XprType>
struct traits<TensorReverseOp<ReverseDimensions,
                              XprType> > : public traits<XprType>
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

template<typename ReverseDimensions, typename XprType>
struct eval<TensorReverseOp<ReverseDimensions, XprType>, Eigen::Dense>
{
  typedef const TensorReverseOp<ReverseDimensions, XprType>& type;
};

template<typename ReverseDimensions, typename XprType>
struct nested<TensorReverseOp<ReverseDimensions, XprType>, 1,
            typename eval<TensorReverseOp<ReverseDimensions, XprType> >::type>
{
  typedef TensorReverseOp<ReverseDimensions, XprType> type;
};

}  // end namespace internal

template<typename ReverseDimensions, typename XprType>
class TensorReverseOp : public TensorBase<TensorReverseOp<ReverseDimensions,
                                          XprType>, WriteAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorReverseOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorReverseOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorReverseOp>::StorageKind
                                                                    StorageKind;
  typedef typename Eigen::internal::traits<TensorReverseOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorReverseOp(
      const XprType& expr, const ReverseDimensions& reverse_dims)
      : m_xpr(expr), m_reverse_dims(reverse_dims) { }

    EIGEN_DEVICE_FUNC
    const ReverseDimensions& reverse() const { return m_reverse_dims; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorReverseOp& operator = (const TensorReverseOp& other)
    {
      typedef TensorAssignOp<TensorReverseOp, const TensorReverseOp> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorReverseOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorReverseOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const ReverseDimensions m_reverse_dims;
};

// Eval as rvalue
template<typename ReverseDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorReverseOp<ReverseDimensions, ArgType>, Device>
{
  typedef TensorReverseOp<ReverseDimensions, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<ReverseDimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = PacketType<CoeffReturnType, Device>::size;
  typedef StorageMemory<CoeffReturnType, Device> Storage;
  typedef typename Storage::Type EvaluatorPointerType;

  enum {
    IsAligned         = false,
    PacketAccess      = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess       = true,
    PreferBlockAccess = true,
    Layout            = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess       = false,  // to be implemented
    RawAccess         = false
  };

  typedef internal::TensorIntDivisor<Index> IndexDivisor;

  typedef typename internal::remove_const<Scalar>::type ScalarNoConst;
  typedef internal::TensorBlock<ScalarNoConst, Index, NumDims, Layout>
      OutputTensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op,
                                                        const Device& device)
      : m_impl(op.expression(), device),
        m_reverse(op.reverse()),
        m_device(device)
  {
    // Reversing a scalar isn't supported yet. It would be a no-op anyway.
    EIGEN_STATIC_ASSERT((NumDims > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);

    // Compute strides
    m_dimensions = m_impl.dimensions();
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_strides[i] = m_strides[i-1] * m_dimensions[i-1];
        if (m_strides[i] > 0) m_fastStrides[i] = IndexDivisor(m_strides[i]);
      }
    } else {
      m_strides[NumDims-1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i+1] * m_dimensions[i+1];
        if (m_strides[i] > 0) m_fastStrides[i] = IndexDivisor(m_strides[i]);
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(EvaluatorPointerType) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index reverseIndex(
      Index index) const {
    eigen_assert(index < dimensions().TotalSize());
    Index inputIndex = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      EIGEN_UNROLL_LOOP
      for (int i = NumDims - 1; i > 0; --i) {
        Index idx = index / m_fastStrides[i];
        index -= idx * m_strides[i];
        if (m_reverse[i]) {
          idx = m_dimensions[i] - idx - 1;
        }
        inputIndex += idx * m_strides[i] ;
      }
      if (m_reverse[0]) {
        inputIndex += (m_dimensions[0] - index - 1);
      } else {
        inputIndex += index;
      }
    } else {
      EIGEN_UNROLL_LOOP
      for (int i = 0; i < NumDims - 1; ++i) {
        Index idx = index / m_fastStrides[i];
        index -= idx * m_strides[i];
        if (m_reverse[i]) {
          idx = m_dimensions[i] - idx - 1;
        }
        inputIndex += idx * m_strides[i] ;
      }
      if (m_reverse[NumDims-1]) {
        inputIndex += (m_dimensions[NumDims-1] - index - 1);
      } else {
        inputIndex += index;
      }
    }
    return inputIndex;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(
      Index index) const  {
    return m_impl.coeff(reverseIndex(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    EIGEN_STATIC_ASSERT((PacketSize > 1), YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+PacketSize-1 < dimensions().TotalSize());

    // TODO(ndjaitly): write a better packing routine that uses
    // local structure.
    EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type
                                                            values[PacketSize];
    EIGEN_UNROLL_LOOP
    for (int i = 0; i < PacketSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
    Eigen::Index block_total_size_max = numext::maxi<Eigen::Index>(
        1, m_device.lastLevelCacheSize() / sizeof(Scalar));
    resources->push_back(internal::TensorOpResourceRequirements(
        internal::kSkewedInnerDims, block_total_size_max));
  }

  struct BlockIteratorState {
    Index block_size;
    Index block_stride;
    Index block_span;
    Index input_size;
    Index input_stride;
    Index input_span;
    Index count;
    bool reverse;
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(
      OutputTensorBlock* output_block) const {
    if (NumDims <= 0) return;

    // TODO(ezhulenev): If underlying tensor expression supports and prefers
    // block evaluation we must use it. Currently we use coeff and packet
    // access into the underlying tensor expression.
    // static const bool useBlockAccessForArgType =
    //     TensorEvaluator<ArgType, Device>::BlockAccess &&
    //     TensorEvaluator<ArgType, Device>::PreferBlockAccess;

    static const bool isColMajor =
        static_cast<int>(Layout) == static_cast<int>(ColMajor);

    static const Index inner_dim_idx = isColMajor ? 0 : NumDims - 1;
    const bool inner_dim_reversed = m_reverse[inner_dim_idx];

    CoeffReturnType* data = output_block->data();
    Index block_offset = 0;

    Index input_offset = reverseIndex(output_block->first_coeff_index());

    // Initialize output block iterator state. Dimension in this array are
    // always in inner_most -> outer_most order (col major layout).
    array<BlockIteratorState, NumDims> it;
    for (Index i = 0; i < NumDims; ++i) {
      const Index dim = isColMajor ? i : NumDims - 1 - i;
      it[i].block_size = output_block->block_sizes()[dim];
      it[i].block_stride = output_block->block_strides()[dim];
      it[i].block_span = it[i].block_stride * (it[i].block_size - 1);
      it[i].input_size = m_dimensions[dim];
      it[i].input_stride = m_strides[dim];
      it[i].input_span = it[i].input_stride * (it[i].input_size - 1);
      it[i].count = 0;
      it[i].reverse = m_reverse[dim];

      if (it[i].reverse) {
        it[i].input_stride = -1 * it[i].input_stride;
        it[i].input_span = -1 * it[i].input_span;
      }
    }

    // If multiple inner dimensions have the same reverse flag, check if we can
    // merge them into a single virtual inner dimension.
    int effective_inner_dim = 0;
    for (int i = 1; i < NumDims; ++i) {
      if (it[i].reverse != it[effective_inner_dim].reverse) break;
      if (it[i].block_stride != it[effective_inner_dim].input_size) break;
      if (it[i].block_stride != numext::abs(it[i].input_stride)) break;

      it[i].block_size = it[effective_inner_dim].block_size * it[i].block_size;
      it[i].input_size = it[effective_inner_dim].input_size * it[i].input_size;

      it[i].block_stride = 1;
      it[i].input_stride = (inner_dim_reversed ? -1 : 1);

      it[i].block_span = it[i].block_stride * (it[i].block_size - 1);
      it[i].input_span = it[i].input_stride * (it[i].input_size - 1);

      effective_inner_dim = i;
    }

    eigen_assert(it[effective_inner_dim].block_stride == 1);
    eigen_assert(it[effective_inner_dim].input_stride ==
                 (inner_dim_reversed ? -1 : 1));

    const Index inner_dim_size = it[effective_inner_dim].block_size;

    while (it[NumDims - 1].count < it[NumDims - 1].block_size) {
      // Copy inner-most dimension data from reversed location in input.
      Index dst = block_offset;
      Index src = input_offset;

      // NOTE(ezhulenev): Adding vectorized path with internal::preverse showed
      // worse results in benchmarks than a simple coefficient loop.
      if (inner_dim_reversed) {
        for (Index i = 0; i < inner_dim_size; ++i) {
          data[dst] = m_impl.coeff(src);
          ++dst;
          --src;
        }
      } else {
        for (Index i = 0; i < inner_dim_size; ++i) {
          data[dst] = m_impl.coeff(src);
          ++dst;
          ++src;
        }
      }

      // For the 1d tensor we need to generate only one inner-most dimension.
      if ((NumDims - effective_inner_dim) == 1) break;

      // Update offset.
      for (Index i = effective_inner_dim + 1; i < NumDims; ++i) {
        if (++it[i].count < it[i].block_size) {
          block_offset += it[i].block_stride;
          input_offset += it[i].input_stride;
          break;
        }
        if (i != NumDims - 1) it[i].count = 0;
        block_offset -= it[i].block_span;
        input_offset -= it[i].input_span;
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    double compute_cost = NumDims * (2 * TensorOpCost::AddCost<Index>() +
                                     2 * TensorOpCost::MulCost<Index>() +
                                     TensorOpCost::DivCost<Index>());
    for (int i = 0; i < NumDims; ++i) {
      if (m_reverse[i]) {
        compute_cost += 2 * TensorOpCost::AddCost<Index>();
      }
    }
    return m_impl.costPerCoeff(vectorized) +
           TensorOpCost(0, 0, compute_cost, false /* vectorized */, PacketSize);
  }

  EIGEN_DEVICE_FUNC typename Storage::Type data() const { return NULL; }

#ifdef EIGEN_USE_SYCL
  // binding placeholder accessors to a command group handler for SYCL
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void bind(cl::sycl::handler &cgh) const {
    m_impl.bind(cgh);
  }
#endif

 protected:
  Dimensions m_dimensions;
  array<Index, NumDims> m_strides;
  array<IndexDivisor, NumDims> m_fastStrides;
  TensorEvaluator<ArgType, Device> m_impl;
  ReverseDimensions m_reverse;
  const Device EIGEN_DEVICE_REF m_device;
};

// Eval as lvalue

template <typename ReverseDimensions, typename ArgType, typename Device>
struct TensorEvaluator<TensorReverseOp<ReverseDimensions, ArgType>, Device>
    : public TensorEvaluator<const TensorReverseOp<ReverseDimensions, ArgType>,
                             Device> {
  typedef TensorEvaluator<const TensorReverseOp<ReverseDimensions, ArgType>,
                          Device> Base;
  typedef TensorReverseOp<ReverseDimensions, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<ReverseDimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = false,
    PreferBlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op,
                                                        const Device& device)
      : Base(op, device) {}

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = PacketType<CoeffReturnType, Device>::size;
  
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Dimensions& dimensions() const { return this->m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    return this->m_impl.coeffRef(this->reverseIndex(index));
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x) {
    EIGEN_STATIC_ASSERT((PacketSize > 1), YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+PacketSize-1 < dimensions().TotalSize());

    // This code is pilfered from TensorMorphing.h
    EIGEN_ALIGN_MAX CoeffReturnType values[PacketSize];
    internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
    EIGEN_UNROLL_LOOP
    for (int i = 0; i < PacketSize; ++i) {
      this->coeffRef(index+i) = values[i];
    }
  }
};


}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REVERSE_H
