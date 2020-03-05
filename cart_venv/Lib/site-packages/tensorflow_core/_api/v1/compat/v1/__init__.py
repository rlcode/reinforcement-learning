# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring in all of the public TensorFlow interface into this module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

from tensorflow.python.tools import module_util as _module_util

# pylint: disable=g-bad-import-order

from tensorflow._api.v1.compat.v1 import app
from tensorflow._api.v1.compat.v1 import audio
from tensorflow._api.v1.compat.v1 import autograph
from tensorflow._api.v1.compat.v1 import bitwise
from tensorflow._api.v1.compat.v1 import compat
from tensorflow._api.v1.compat.v1 import config
from tensorflow._api.v1.compat.v1 import data
from tensorflow._api.v1.compat.v1 import debugging
from tensorflow._api.v1.compat.v1 import distribute
from tensorflow._api.v1.compat.v1 import distributions
from tensorflow._api.v1.compat.v1 import dtypes
from tensorflow._api.v1.compat.v1 import errors
from tensorflow._api.v1.compat.v1 import experimental
from tensorflow._api.v1.compat.v1 import feature_column
from tensorflow._api.v1.compat.v1 import gfile
from tensorflow._api.v1.compat.v1 import graph_util
from tensorflow._api.v1.compat.v1 import image
from tensorflow._api.v1.compat.v1 import initializers
from tensorflow._api.v1.compat.v1 import io
from tensorflow._api.v1.compat.v1 import layers
from tensorflow._api.v1.compat.v1 import linalg
from tensorflow._api.v1.compat.v1 import lite
from tensorflow._api.v1.compat.v1 import logging
from tensorflow._api.v1.compat.v1 import lookup
from tensorflow._api.v1.compat.v1 import losses
from tensorflow._api.v1.compat.v1 import manip
from tensorflow._api.v1.compat.v1 import math
from tensorflow._api.v1.compat.v1 import metrics
from tensorflow._api.v1.compat.v1 import nest
from tensorflow._api.v1.compat.v1 import nn
from tensorflow._api.v1.compat.v1 import profiler
from tensorflow._api.v1.compat.v1 import python_io
from tensorflow._api.v1.compat.v1 import quantization
from tensorflow._api.v1.compat.v1 import queue
from tensorflow._api.v1.compat.v1 import ragged
from tensorflow._api.v1.compat.v1 import random
from tensorflow._api.v1.compat.v1 import raw_ops
from tensorflow._api.v1.compat.v1 import resource_loader
from tensorflow._api.v1.compat.v1 import saved_model
from tensorflow._api.v1.compat.v1 import sets
from tensorflow._api.v1.compat.v1 import signal
from tensorflow._api.v1.compat.v1 import sparse
from tensorflow._api.v1.compat.v1 import spectral
from tensorflow._api.v1.compat.v1 import strings
from tensorflow._api.v1.compat.v1 import summary
from tensorflow._api.v1.compat.v1 import sysconfig
from tensorflow._api.v1.compat.v1 import test
from tensorflow._api.v1.compat.v1 import tpu
from tensorflow._api.v1.compat.v1 import train
from tensorflow._api.v1.compat.v1 import user_ops
from tensorflow._api.v1.compat.v1 import version
from tensorflow._api.v1.compat.v1 import xla
from tensorflow.python import AttrValue
from tensorflow.python import ConfigProto
from tensorflow.python import Event
from tensorflow.python import GPUOptions
from tensorflow.python import GraphDef
from tensorflow.python import GraphOptions
from tensorflow.python import HistogramProto
from tensorflow.python import LogMessage
from tensorflow.python import MetaGraphDef
from tensorflow.python import NameAttrList
from tensorflow.python import NodeDef
from tensorflow.python import OptimizerOptions
from tensorflow.python import RunMetadata
from tensorflow.python import RunOptions
from tensorflow.python import SessionLog
from tensorflow.python import Summary
from tensorflow.python import SummaryMetadata
from tensorflow.python import TensorInfo
from tensorflow.python.client.session import InteractiveSession
from tensorflow.python.client.session import Session
from tensorflow.python.compat.v2_compat import disable_v2_behavior
from tensorflow.python.compat.v2_compat import enable_v2_behavior
from tensorflow.python.data.ops.optional_ops import OptionalSpec
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.def_function import function
from tensorflow.python.eager.wrap_function import wrap_function
from tensorflow.python.framework.constant_op import constant_v1 as constant
from tensorflow.python.framework.device_spec import DeviceSpecV1 as DeviceSpec
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.framework.dtypes import QUANTIZED_DTYPES
from tensorflow.python.framework.dtypes import as_dtype
from tensorflow.python.framework.dtypes import bfloat16
from tensorflow.python.framework.dtypes import bool
from tensorflow.python.framework.dtypes import complex128
from tensorflow.python.framework.dtypes import complex64
from tensorflow.python.framework.dtypes import double
from tensorflow.python.framework.dtypes import float16
from tensorflow.python.framework.dtypes import float32
from tensorflow.python.framework.dtypes import float64
from tensorflow.python.framework.dtypes import half
from tensorflow.python.framework.dtypes import int16
from tensorflow.python.framework.dtypes import int32
from tensorflow.python.framework.dtypes import int64
from tensorflow.python.framework.dtypes import int8
from tensorflow.python.framework.dtypes import qint16
from tensorflow.python.framework.dtypes import qint32
from tensorflow.python.framework.dtypes import qint8
from tensorflow.python.framework.dtypes import quint16
from tensorflow.python.framework.dtypes import quint8
from tensorflow.python.framework.dtypes import resource
from tensorflow.python.framework.dtypes import string
from tensorflow.python.framework.dtypes import uint16
from tensorflow.python.framework.dtypes import uint32
from tensorflow.python.framework.dtypes import uint64
from tensorflow.python.framework.dtypes import uint8
from tensorflow.python.framework.dtypes import variant
from tensorflow.python.framework.errors_impl import OpError
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.framework.indexed_slices import IndexedSlicesSpec
from tensorflow.python.framework.indexed_slices import convert_to_tensor_or_indexed_slices
from tensorflow.python.framework.load_library import load_file_system_library
from tensorflow.python.framework.load_library import load_library
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import NoGradient
from tensorflow.python.framework.ops import NoGradient as NotDifferentiable
from tensorflow.python.framework.ops import NoGradient as no_gradient
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import RegisterGradient
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import _colocate_with as colocate_with
from tensorflow.python.framework.ops import add_to_collection
from tensorflow.python.framework.ops import add_to_collections
from tensorflow.python.framework.ops import container
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.ops import device
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import disable_tensor_equality
from tensorflow.python.framework.ops import enable_eager_execution
from tensorflow.python.framework.ops import enable_tensor_equality
from tensorflow.python.framework.ops import get_collection
from tensorflow.python.framework.ops import get_collection_ref
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.framework.ops import get_default_session
from tensorflow.python.framework.ops import init_scope
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.ops import op_scope
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.framework.random_seed import get_seed
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec
from tensorflow.python.framework.sparse_tensor import SparseTensorValue
from tensorflow.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_shape import dimension_at_index
from tensorflow.python.framework.tensor_shape import dimension_value
from tensorflow.python.framework.tensor_shape import disable_v2_tensorshape
from tensorflow.python.framework.tensor_shape import enable_v2_tensorshape
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray
from tensorflow.python.framework.tensor_util import constant_value as get_static_value
from tensorflow.python.framework.tensor_util import is_tensor
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.python.framework.type_spec import TypeSpec
from tensorflow.python.framework.versions import COMPILER_VERSION
from tensorflow.python.framework.versions import COMPILER_VERSION as __compiler_version__
from tensorflow.python.framework.versions import CXX11_ABI_FLAG
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as __cxx11_abi_flag__
from tensorflow.python.framework.versions import GIT_VERSION
from tensorflow.python.framework.versions import GIT_VERSION as __git_version__
from tensorflow.python.framework.versions import GRAPH_DEF_VERSION
from tensorflow.python.framework.versions import GRAPH_DEF_VERSION_MIN_CONSUMER
from tensorflow.python.framework.versions import GRAPH_DEF_VERSION_MIN_PRODUCER
from tensorflow.python.framework.versions import MONOLITHIC_BUILD
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as __monolithic_build__
from tensorflow.python.framework.versions import VERSION
from tensorflow.python.framework.versions import VERSION as __version__
from tensorflow.python.module.module import Module
from tensorflow.python.ops.array_ops import batch_gather
from tensorflow.python.ops.array_ops import batch_to_space
from tensorflow.python.ops.array_ops import boolean_mask
from tensorflow.python.ops.array_ops import broadcast_dynamic_shape
from tensorflow.python.ops.array_ops import broadcast_static_shape
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.array_ops import depth_to_space
from tensorflow.python.ops.array_ops import edit_distance
from tensorflow.python.ops.array_ops import expand_dims
from tensorflow.python.ops.array_ops import extract_image_patches
from tensorflow.python.ops.array_ops import fill
from tensorflow.python.ops.array_ops import fingerprint
from tensorflow.python.ops.array_ops import gather
from tensorflow.python.ops.array_ops import gather_nd
from tensorflow.python.ops.array_ops import identity
from tensorflow.python.ops.array_ops import matrix_diag
from tensorflow.python.ops.array_ops import matrix_diag_part
from tensorflow.python.ops.array_ops import matrix_set_diag
from tensorflow.python.ops.array_ops import matrix_transpose
from tensorflow.python.ops.array_ops import meshgrid
from tensorflow.python.ops.array_ops import newaxis
from tensorflow.python.ops.array_ops import one_hot
from tensorflow.python.ops.array_ops import ones
from tensorflow.python.ops.array_ops import ones_like
from tensorflow.python.ops.array_ops import pad
from tensorflow.python.ops.array_ops import parallel_stack
from tensorflow.python.ops.array_ops import placeholder
from tensorflow.python.ops.array_ops import placeholder_with_default
from tensorflow.python.ops.array_ops import quantize
from tensorflow.python.ops.array_ops import quantize_v2
from tensorflow.python.ops.array_ops import rank
from tensorflow.python.ops.array_ops import repeat
from tensorflow.python.ops.array_ops import required_space_to_batch_paddings
from tensorflow.python.ops.array_ops import reshape
from tensorflow.python.ops.array_ops import reverse_sequence
from tensorflow.python.ops.array_ops import searchsorted
from tensorflow.python.ops.array_ops import sequence_mask
from tensorflow.python.ops.array_ops import setdiff1d
from tensorflow.python.ops.array_ops import shape
from tensorflow.python.ops.array_ops import shape_n
from tensorflow.python.ops.array_ops import size
from tensorflow.python.ops.array_ops import slice
from tensorflow.python.ops.array_ops import space_to_batch
from tensorflow.python.ops.array_ops import space_to_depth
from tensorflow.python.ops.array_ops import sparse_mask
from tensorflow.python.ops.array_ops import sparse_placeholder
from tensorflow.python.ops.array_ops import split
from tensorflow.python.ops.array_ops import squeeze
from tensorflow.python.ops.array_ops import stack
from tensorflow.python.ops.array_ops import strided_slice
from tensorflow.python.ops.array_ops import transpose
from tensorflow.python.ops.array_ops import unique
from tensorflow.python.ops.array_ops import unique_with_counts
from tensorflow.python.ops.array_ops import unstack
from tensorflow.python.ops.array_ops import where
from tensorflow.python.ops.array_ops import where_v2
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.array_ops import zeros_like
from tensorflow.python.ops.batch_ops import batch_function as nondifferentiable_batch_function
from tensorflow.python.ops.check_ops import assert_equal
from tensorflow.python.ops.check_ops import assert_greater
from tensorflow.python.ops.check_ops import assert_greater_equal
from tensorflow.python.ops.check_ops import assert_integer
from tensorflow.python.ops.check_ops import assert_less
from tensorflow.python.ops.check_ops import assert_less_equal
from tensorflow.python.ops.check_ops import assert_near
from tensorflow.python.ops.check_ops import assert_negative
from tensorflow.python.ops.check_ops import assert_non_negative
from tensorflow.python.ops.check_ops import assert_non_positive
from tensorflow.python.ops.check_ops import assert_none_equal
from tensorflow.python.ops.check_ops import assert_positive
from tensorflow.python.ops.check_ops import assert_proper_iterable
from tensorflow.python.ops.check_ops import assert_rank
from tensorflow.python.ops.check_ops import assert_rank_at_least
from tensorflow.python.ops.check_ops import assert_rank_in
from tensorflow.python.ops.check_ops import assert_same_float_dtype
from tensorflow.python.ops.check_ops import assert_scalar
from tensorflow.python.ops.check_ops import assert_type
from tensorflow.python.ops.check_ops import ensure_shape
from tensorflow.python.ops.check_ops import is_non_decreasing
from tensorflow.python.ops.check_ops import is_numeric_tensor
from tensorflow.python.ops.check_ops import is_strictly_increasing
from tensorflow.python.ops.clip_ops import clip_by_average_norm
from tensorflow.python.ops.clip_ops import clip_by_global_norm
from tensorflow.python.ops.clip_ops import clip_by_norm
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.clip_ops import global_norm
from tensorflow.python.ops.confusion_matrix import confusion_matrix_v1 as confusion_matrix
from tensorflow.python.ops.control_flow_ops import Assert
from tensorflow.python.ops.control_flow_ops import case
from tensorflow.python.ops.control_flow_ops import cond
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.control_flow_ops import switch_case
from tensorflow.python.ops.control_flow_ops import tuple
from tensorflow.python.ops.control_flow_ops import while_loop
from tensorflow.python.ops.control_flow_v2_toggles import control_flow_v2_enabled
from tensorflow.python.ops.control_flow_v2_toggles import disable_control_flow_v2
from tensorflow.python.ops.control_flow_v2_toggles import enable_control_flow_v2
from tensorflow.python.ops.critical_section_ops import CriticalSection
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.ops.custom_gradient import grad_pass_through
from tensorflow.python.ops.custom_gradient import recompute_grad
from tensorflow.python.ops.data_flow_ops import ConditionalAccumulator
from tensorflow.python.ops.data_flow_ops import ConditionalAccumulatorBase
from tensorflow.python.ops.data_flow_ops import FIFOQueue
from tensorflow.python.ops.data_flow_ops import PaddingFIFOQueue
from tensorflow.python.ops.data_flow_ops import PriorityQueue
from tensorflow.python.ops.data_flow_ops import QueueBase
from tensorflow.python.ops.data_flow_ops import RandomShuffleQueue
from tensorflow.python.ops.data_flow_ops import SparseConditionalAccumulator
from tensorflow.python.ops.functional_ops import foldl
from tensorflow.python.ops.functional_ops import foldr
from tensorflow.python.ops.functional_ops import scan
from tensorflow.python.ops.gen_array_ops import batch_to_space_nd
from tensorflow.python.ops.gen_array_ops import bitcast
from tensorflow.python.ops.gen_array_ops import broadcast_to
from tensorflow.python.ops.gen_array_ops import check_numerics
from tensorflow.python.ops.gen_array_ops import dequantize
from tensorflow.python.ops.gen_array_ops import diag
from tensorflow.python.ops.gen_array_ops import diag_part
from tensorflow.python.ops.gen_array_ops import extract_volume_patches
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_args
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_args_gradient
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_vars
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_vars_gradient
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_vars_per_channel
from tensorflow.python.ops.gen_array_ops import fake_quant_with_min_max_vars_per_channel_gradient
from tensorflow.python.ops.gen_array_ops import guarantee_const
from tensorflow.python.ops.gen_array_ops import identity_n
from tensorflow.python.ops.gen_array_ops import invert_permutation
from tensorflow.python.ops.gen_array_ops import matrix_band_part
from tensorflow.python.ops.gen_array_ops import quantized_concat
from tensorflow.python.ops.gen_array_ops import reverse_v2
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse
from tensorflow.python.ops.gen_array_ops import scatter_nd
from tensorflow.python.ops.gen_array_ops import space_to_batch_nd
from tensorflow.python.ops.gen_array_ops import stop_gradient
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add as tensor_scatter_nd_add
from tensorflow.python.ops.gen_array_ops import tensor_scatter_sub
from tensorflow.python.ops.gen_array_ops import tensor_scatter_sub as tensor_scatter_nd_sub
from tensorflow.python.ops.gen_array_ops import tensor_scatter_update
from tensorflow.python.ops.gen_array_ops import tensor_scatter_update as tensor_scatter_nd_update
from tensorflow.python.ops.gen_array_ops import tile
from tensorflow.python.ops.gen_array_ops import unravel_index
from tensorflow.python.ops.gen_control_flow_ops import no_op
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition
from tensorflow.python.ops.gen_data_flow_ops import dynamic_stitch
from tensorflow.python.ops.gen_io_ops import matching_files
from tensorflow.python.ops.gen_io_ops import read_file
from tensorflow.python.ops.gen_io_ops import write_file
from tensorflow.python.ops.gen_linalg_ops import cholesky
from tensorflow.python.ops.gen_linalg_ops import matrix_determinant
from tensorflow.python.ops.gen_linalg_ops import matrix_inverse
from tensorflow.python.ops.gen_linalg_ops import matrix_solve
from tensorflow.python.ops.gen_linalg_ops import matrix_square_root
from tensorflow.python.ops.gen_linalg_ops import matrix_triangular_solve
from tensorflow.python.ops.gen_linalg_ops import qr
from tensorflow.python.ops.gen_logging_ops import timestamp
from tensorflow.python.ops.gen_math_ops import acos
from tensorflow.python.ops.gen_math_ops import acosh
from tensorflow.python.ops.gen_math_ops import add
from tensorflow.python.ops.gen_math_ops import arg_max
from tensorflow.python.ops.gen_math_ops import arg_min
from tensorflow.python.ops.gen_math_ops import asin
from tensorflow.python.ops.gen_math_ops import asinh
from tensorflow.python.ops.gen_math_ops import atan
from tensorflow.python.ops.gen_math_ops import atan2
from tensorflow.python.ops.gen_math_ops import atanh
from tensorflow.python.ops.gen_math_ops import betainc
from tensorflow.python.ops.gen_math_ops import ceil
from tensorflow.python.ops.gen_math_ops import cos
from tensorflow.python.ops.gen_math_ops import cosh
from tensorflow.python.ops.gen_math_ops import cross
from tensorflow.python.ops.gen_math_ops import digamma
from tensorflow.python.ops.gen_math_ops import erf
from tensorflow.python.ops.gen_math_ops import erfc
from tensorflow.python.ops.gen_math_ops import exp
from tensorflow.python.ops.gen_math_ops import expm1
from tensorflow.python.ops.gen_math_ops import floor
from tensorflow.python.ops.gen_math_ops import floor_div
from tensorflow.python.ops.gen_math_ops import floor_mod as floormod
from tensorflow.python.ops.gen_math_ops import floor_mod as mod
from tensorflow.python.ops.gen_math_ops import greater
from tensorflow.python.ops.gen_math_ops import greater_equal
from tensorflow.python.ops.gen_math_ops import igamma
from tensorflow.python.ops.gen_math_ops import igammac
from tensorflow.python.ops.gen_math_ops import is_finite
from tensorflow.python.ops.gen_math_ops import is_inf
from tensorflow.python.ops.gen_math_ops import is_nan
from tensorflow.python.ops.gen_math_ops import less
from tensorflow.python.ops.gen_math_ops import less_equal
from tensorflow.python.ops.gen_math_ops import lgamma
from tensorflow.python.ops.gen_math_ops import lin_space
from tensorflow.python.ops.gen_math_ops import lin_space as linspace
from tensorflow.python.ops.gen_math_ops import log
from tensorflow.python.ops.gen_math_ops import log1p
from tensorflow.python.ops.gen_math_ops import logical_and
from tensorflow.python.ops.gen_math_ops import logical_not
from tensorflow.python.ops.gen_math_ops import logical_or
from tensorflow.python.ops.gen_math_ops import maximum
from tensorflow.python.ops.gen_math_ops import minimum
from tensorflow.python.ops.gen_math_ops import neg as negative
from tensorflow.python.ops.gen_math_ops import polygamma
from tensorflow.python.ops.gen_math_ops import real_div as realdiv
from tensorflow.python.ops.gen_math_ops import reciprocal
from tensorflow.python.ops.gen_math_ops import rint
from tensorflow.python.ops.gen_math_ops import rsqrt
from tensorflow.python.ops.gen_math_ops import segment_max
from tensorflow.python.ops.gen_math_ops import segment_mean
from tensorflow.python.ops.gen_math_ops import segment_min
from tensorflow.python.ops.gen_math_ops import segment_prod
from tensorflow.python.ops.gen_math_ops import segment_sum
from tensorflow.python.ops.gen_math_ops import sign
from tensorflow.python.ops.gen_math_ops import sin
from tensorflow.python.ops.gen_math_ops import sinh
from tensorflow.python.ops.gen_math_ops import sparse_mat_mul as sparse_matmul
from tensorflow.python.ops.gen_math_ops import sqrt
from tensorflow.python.ops.gen_math_ops import square
from tensorflow.python.ops.gen_math_ops import squared_difference
from tensorflow.python.ops.gen_math_ops import tan
from tensorflow.python.ops.gen_math_ops import tanh
from tensorflow.python.ops.gen_math_ops import truncate_div as truncatediv
from tensorflow.python.ops.gen_math_ops import truncate_mod as truncatemod
from tensorflow.python.ops.gen_math_ops import unsorted_segment_max
from tensorflow.python.ops.gen_math_ops import unsorted_segment_min
from tensorflow.python.ops.gen_math_ops import unsorted_segment_prod
from tensorflow.python.ops.gen_math_ops import unsorted_segment_sum
from tensorflow.python.ops.gen_math_ops import zeta
from tensorflow.python.ops.gen_parsing_ops import decode_compressed
from tensorflow.python.ops.gen_parsing_ops import decode_json_example
from tensorflow.python.ops.gen_parsing_ops import parse_tensor
from tensorflow.python.ops.gen_parsing_ops import serialize_tensor
from tensorflow.python.ops.gen_spectral_ops import fft
from tensorflow.python.ops.gen_spectral_ops import fft2d
from tensorflow.python.ops.gen_spectral_ops import fft3d
from tensorflow.python.ops.gen_spectral_ops import ifft
from tensorflow.python.ops.gen_spectral_ops import ifft2d
from tensorflow.python.ops.gen_spectral_ops import ifft3d
from tensorflow.python.ops.gen_string_ops import as_string
from tensorflow.python.ops.gen_string_ops import decode_base64
from tensorflow.python.ops.gen_string_ops import encode_base64
from tensorflow.python.ops.gen_string_ops import string_join
from tensorflow.python.ops.gen_string_ops import string_strip
from tensorflow.python.ops.gen_string_ops import string_to_hash_bucket_fast
from tensorflow.python.ops.gen_string_ops import string_to_hash_bucket_strong
from tensorflow.python.ops.gradients_impl import gradients
from tensorflow.python.ops.gradients_impl import hessians
from tensorflow.python.ops.gradients_util import AggregationMethod
from tensorflow.python.ops.histogram_ops import histogram_fixed_width
from tensorflow.python.ops.histogram_ops import histogram_fixed_width_bins
from tensorflow.python.ops.init_ops import Constant as constant_initializer
from tensorflow.python.ops.init_ops import GlorotNormal as glorot_normal_initializer
from tensorflow.python.ops.init_ops import GlorotUniform as glorot_uniform_initializer
from tensorflow.python.ops.init_ops import Ones as ones_initializer
from tensorflow.python.ops.init_ops import Orthogonal as orthogonal_initializer
from tensorflow.python.ops.init_ops import RandomNormal as random_normal_initializer
from tensorflow.python.ops.init_ops import RandomUniform as random_uniform_initializer
from tensorflow.python.ops.init_ops import TruncatedNormal as truncated_normal_initializer
from tensorflow.python.ops.init_ops import UniformUnitScaling as uniform_unit_scaling_initializer
from tensorflow.python.ops.init_ops import VarianceScaling as variance_scaling_initializer
from tensorflow.python.ops.init_ops import Zeros as zeros_initializer
from tensorflow.python.ops.io_ops import FixedLengthRecordReader
from tensorflow.python.ops.io_ops import IdentityReader
from tensorflow.python.ops.io_ops import LMDBReader
from tensorflow.python.ops.io_ops import ReaderBase
from tensorflow.python.ops.io_ops import TFRecordReader
from tensorflow.python.ops.io_ops import TextLineReader
from tensorflow.python.ops.io_ops import WholeFileReader
from tensorflow.python.ops.linalg_ops import cholesky_solve
from tensorflow.python.ops.linalg_ops import eye
from tensorflow.python.ops.linalg_ops import matrix_solve_ls
from tensorflow.python.ops.linalg_ops import norm
from tensorflow.python.ops.linalg_ops import self_adjoint_eig
from tensorflow.python.ops.linalg_ops import self_adjoint_eigvals
from tensorflow.python.ops.linalg_ops import svd
from tensorflow.python.ops.logging_ops import Print
from tensorflow.python.ops.logging_ops import print_v2 as print
from tensorflow.python.ops.lookup_ops import initialize_all_tables
from tensorflow.python.ops.lookup_ops import tables_initializer
from tensorflow.python.ops.manip_ops import roll
from tensorflow.python.ops.map_fn import map_fn
from tensorflow.python.ops.math_ops import abs
from tensorflow.python.ops.math_ops import accumulate_n
from tensorflow.python.ops.math_ops import add_n
from tensorflow.python.ops.math_ops import angle
from tensorflow.python.ops.math_ops import argmax
from tensorflow.python.ops.math_ops import argmin
from tensorflow.python.ops.math_ops import bincount_v1 as bincount
from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.math_ops import complex
from tensorflow.python.ops.math_ops import conj
from tensorflow.python.ops.math_ops import count_nonzero
from tensorflow.python.ops.math_ops import cumprod
from tensorflow.python.ops.math_ops import cumsum
from tensorflow.python.ops.math_ops import div
from tensorflow.python.ops.math_ops import div_no_nan
from tensorflow.python.ops.math_ops import divide
from tensorflow.python.ops.math_ops import equal
from tensorflow.python.ops.math_ops import floordiv
from tensorflow.python.ops.math_ops import imag
from tensorflow.python.ops.math_ops import log_sigmoid
from tensorflow.python.ops.math_ops import logical_xor
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.math_ops import multiply
from tensorflow.python.ops.math_ops import not_equal
from tensorflow.python.ops.math_ops import pow
from tensorflow.python.ops.math_ops import range
from tensorflow.python.ops.math_ops import real
from tensorflow.python.ops.math_ops import reduce_all_v1 as reduce_all
from tensorflow.python.ops.math_ops import reduce_any_v1 as reduce_any
from tensorflow.python.ops.math_ops import reduce_logsumexp_v1 as reduce_logsumexp
from tensorflow.python.ops.math_ops import reduce_max_v1 as reduce_max
from tensorflow.python.ops.math_ops import reduce_mean_v1 as reduce_mean
from tensorflow.python.ops.math_ops import reduce_min_v1 as reduce_min
from tensorflow.python.ops.math_ops import reduce_prod_v1 as reduce_prod
from tensorflow.python.ops.math_ops import reduce_sum_v1 as reduce_sum
from tensorflow.python.ops.math_ops import round
from tensorflow.python.ops.math_ops import saturate_cast
from tensorflow.python.ops.math_ops import scalar_mul
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import sparse_segment_mean
from tensorflow.python.ops.math_ops import sparse_segment_sqrt_n
from tensorflow.python.ops.math_ops import sparse_segment_sum
from tensorflow.python.ops.math_ops import subtract
from tensorflow.python.ops.math_ops import tensordot
from tensorflow.python.ops.math_ops import to_bfloat16
from tensorflow.python.ops.math_ops import to_complex128
from tensorflow.python.ops.math_ops import to_complex64
from tensorflow.python.ops.math_ops import to_double
from tensorflow.python.ops.math_ops import to_float
from tensorflow.python.ops.math_ops import to_int32
from tensorflow.python.ops.math_ops import to_int64
from tensorflow.python.ops.math_ops import trace
from tensorflow.python.ops.math_ops import truediv
from tensorflow.python.ops.math_ops import unsorted_segment_mean
from tensorflow.python.ops.math_ops import unsorted_segment_sqrt_n
from tensorflow.python.ops.numerics import add_check_numerics_ops
from tensorflow.python.ops.numerics import verify_tensor_all_finite
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.ops.parsing_ops import FixedLenFeature
from tensorflow.python.ops.parsing_ops import FixedLenSequenceFeature
from tensorflow.python.ops.parsing_ops import SparseFeature
from tensorflow.python.ops.parsing_ops import VarLenFeature
from tensorflow.python.ops.parsing_ops import decode_csv
from tensorflow.python.ops.parsing_ops import decode_raw_v1 as decode_raw
from tensorflow.python.ops.parsing_ops import parse_example
from tensorflow.python.ops.parsing_ops import parse_single_example
from tensorflow.python.ops.parsing_ops import parse_single_sequence_example
from tensorflow.python.ops.partitioned_variables import create_partitioned_variables
from tensorflow.python.ops.partitioned_variables import fixed_size_partitioner
from tensorflow.python.ops.partitioned_variables import min_max_variable_partitioner
from tensorflow.python.ops.partitioned_variables import variable_axis_size_partitioner
from tensorflow.python.ops.ragged.ragged_string_ops import string_split
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.random_ops import multinomial
from tensorflow.python.ops.random_ops import random_crop
from tensorflow.python.ops.random_ops import random_gamma
from tensorflow.python.ops.random_ops import random_normal
from tensorflow.python.ops.random_ops import random_poisson
from tensorflow.python.ops.random_ops import random_shuffle
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.ops.random_ops import truncated_normal
from tensorflow.python.ops.script_ops import eager_py_func as py_function
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.ops.script_ops import py_func
from tensorflow.python.ops.session_ops import delete_session_tensor
from tensorflow.python.ops.session_ops import get_session_handle
from tensorflow.python.ops.session_ops import get_session_tensor
from tensorflow.python.ops.sort_ops import argsort
from tensorflow.python.ops.sort_ops import sort
from tensorflow.python.ops.sparse_ops import deserialize_many_sparse
from tensorflow.python.ops.sparse_ops import serialize_many_sparse
from tensorflow.python.ops.sparse_ops import serialize_sparse
from tensorflow.python.ops.sparse_ops import sparse_add
from tensorflow.python.ops.sparse_ops import sparse_concat
from tensorflow.python.ops.sparse_ops import sparse_fill_empty_rows
from tensorflow.python.ops.sparse_ops import sparse_maximum
from tensorflow.python.ops.sparse_ops import sparse_merge
from tensorflow.python.ops.sparse_ops import sparse_minimum
from tensorflow.python.ops.sparse_ops import sparse_reduce_max
from tensorflow.python.ops.sparse_ops import sparse_reduce_max_sparse
from tensorflow.python.ops.sparse_ops import sparse_reduce_sum
from tensorflow.python.ops.sparse_ops import sparse_reduce_sum_sparse
from tensorflow.python.ops.sparse_ops import sparse_reorder
from tensorflow.python.ops.sparse_ops import sparse_reset_shape
from tensorflow.python.ops.sparse_ops import sparse_reshape
from tensorflow.python.ops.sparse_ops import sparse_retain
from tensorflow.python.ops.sparse_ops import sparse_slice
from tensorflow.python.ops.sparse_ops import sparse_softmax
from tensorflow.python.ops.sparse_ops import sparse_split
from tensorflow.python.ops.sparse_ops import sparse_tensor_dense_matmul
from tensorflow.python.ops.sparse_ops import sparse_tensor_to_dense
from tensorflow.python.ops.sparse_ops import sparse_to_dense
from tensorflow.python.ops.sparse_ops import sparse_to_indicator
from tensorflow.python.ops.sparse_ops import sparse_transpose
from tensorflow.python.ops.special_math_ops import einsum
from tensorflow.python.ops.special_math_ops import lbeta
from tensorflow.python.ops.state_ops import assign
from tensorflow.python.ops.state_ops import assign_add
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.state_ops import batch_scatter_update
from tensorflow.python.ops.state_ops import count_up_to
from tensorflow.python.ops.state_ops import scatter_add
from tensorflow.python.ops.state_ops import scatter_div
from tensorflow.python.ops.state_ops import scatter_max
from tensorflow.python.ops.state_ops import scatter_min
from tensorflow.python.ops.state_ops import scatter_mul
from tensorflow.python.ops.state_ops import scatter_nd_add
from tensorflow.python.ops.state_ops import scatter_nd_sub
from tensorflow.python.ops.state_ops import scatter_nd_update
from tensorflow.python.ops.state_ops import scatter_sub
from tensorflow.python.ops.state_ops import scatter_update
from tensorflow.python.ops.string_ops import reduce_join
from tensorflow.python.ops.string_ops import regex_replace
from tensorflow.python.ops.string_ops import string_to_hash_bucket_v1 as string_to_hash_bucket
from tensorflow.python.ops.string_ops import string_to_number_v1 as string_to_number
from tensorflow.python.ops.string_ops import substr_deprecated as substr
from tensorflow.python.ops.template import make_template
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.ops.variable_scope import AUTO_REUSE
from tensorflow.python.ops.variable_scope import VariableScope
from tensorflow.python.ops.variable_scope import disable_resource_variables
from tensorflow.python.ops.variable_scope import enable_resource_variables
from tensorflow.python.ops.variable_scope import get_local_variable
from tensorflow.python.ops.variable_scope import get_variable
from tensorflow.python.ops.variable_scope import get_variable_scope
from tensorflow.python.ops.variable_scope import no_regularizer
from tensorflow.python.ops.variable_scope import resource_variables_enabled
from tensorflow.python.ops.variable_scope import variable_creator_scope_v1 as variable_creator_scope
from tensorflow.python.ops.variable_scope import variable_op_scope
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.ops.variables import VariableAggregation
from tensorflow.python.ops.variables import VariableSynchronization
from tensorflow.python.ops.variables import VariableV1 as Variable
from tensorflow.python.ops.variables import all_variables
from tensorflow.python.ops.variables import assert_variables_initialized
from tensorflow.python.ops.variables import global_variables
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.ops.variables import initialize_all_variables
from tensorflow.python.ops.variables import initialize_local_variables
from tensorflow.python.ops.variables import initialize_variables
from tensorflow.python.ops.variables import is_variable_initialized
from tensorflow.python.ops.variables import local_variables
from tensorflow.python.ops.variables import local_variables_initializer
from tensorflow.python.ops.variables import model_variables
from tensorflow.python.ops.variables import moving_average_variables
from tensorflow.python.ops.variables import report_uninitialized_variables
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.python.ops.variables import variables_initializer
from tensorflow.python.platform.tf_logging import get_logger


from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v1", public_apis=None, deprecation=False,
      has_lite=False)


# Hook external TensorFlow modules.
_current_module = _sys.modules[__name__]
try:
  from tensorflow_estimator.python.estimator.api._v1 import estimator
  _current_module.__path__ = (
      [_module_util.get_parent_dir(estimator)] + _current_module.__path__)
  setattr(_current_module, "estimator", estimator)
except ImportError:
  pass

try:
  from tensorflow.python.keras.api._v1 import keras
  _current_module.__path__ = (
      [_module_util.get_parent_dir(keras)] + _current_module.__path__)
  setattr(_current_module, "keras", keras)
except ImportError:
  pass


from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)
