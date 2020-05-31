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

import logging as _logging
import os as _os
import sys as _sys

from tensorflow.python.tools import module_util as _module_util

# pylint: disable=g-bad-import-order

from tensorflow._api.v1.compat.v2 import audio
from tensorflow._api.v1.compat.v2 import autograph
from tensorflow._api.v1.compat.v2 import bitwise
from tensorflow._api.v1.compat.v2 import compat
from tensorflow._api.v1.compat.v2 import config
from tensorflow._api.v1.compat.v2 import data
from tensorflow._api.v1.compat.v2 import debugging
from tensorflow._api.v1.compat.v2 import distribute
from tensorflow._api.v1.compat.v2 import dtypes
from tensorflow._api.v1.compat.v2 import errors
from tensorflow._api.v1.compat.v2 import experimental
from tensorflow._api.v1.compat.v2 import feature_column
from tensorflow._api.v1.compat.v2 import graph_util
from tensorflow._api.v1.compat.v2 import image
from tensorflow._api.v1.compat.v2 import io
from tensorflow._api.v1.compat.v2 import linalg
from tensorflow._api.v1.compat.v2 import lite
from tensorflow._api.v1.compat.v2 import lookup
from tensorflow._api.v1.compat.v2 import math
from tensorflow._api.v1.compat.v2 import nest
from tensorflow._api.v1.compat.v2 import nn
from tensorflow._api.v1.compat.v2 import quantization
from tensorflow._api.v1.compat.v2 import queue
from tensorflow._api.v1.compat.v2 import ragged
from tensorflow._api.v1.compat.v2 import random
from tensorflow._api.v1.compat.v2 import raw_ops
from tensorflow._api.v1.compat.v2 import saved_model
from tensorflow._api.v1.compat.v2 import sets
from tensorflow._api.v1.compat.v2 import signal
from tensorflow._api.v1.compat.v2 import sparse
from tensorflow._api.v1.compat.v2 import strings
from tensorflow._api.v1.compat.v2 import summary
from tensorflow._api.v1.compat.v2 import sysconfig
from tensorflow._api.v1.compat.v2 import test
from tensorflow._api.v1.compat.v2 import tpu
from tensorflow._api.v1.compat.v2 import train
from tensorflow._api.v1.compat.v2 import version
from tensorflow._api.v1.compat.v2 import xla
from tensorflow.python.data.ops.optional_ops import OptionalSpec
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.device_spec import DeviceSpecV2 as DeviceSpec
from tensorflow.python.framework.dtypes import DType
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
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.framework.indexed_slices import IndexedSlicesSpec
from tensorflow.python.framework.load_library import load_library
from tensorflow.python.framework.load_library import load_op_library
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.ops import NoGradient as no_gradient
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import RegisterGradient
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.ops import convert_to_tensor_v2 as convert_to_tensor
from tensorflow.python.framework.ops import device_v2 as device
from tensorflow.python.framework.ops import init_scope
from tensorflow.python.framework.ops import name_scope_v2 as name_scope
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorSpec
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray
from tensorflow.python.framework.tensor_util import constant_value as get_static_value
from tensorflow.python.framework.tensor_util import is_tensor
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.python.framework.type_spec import TypeSpec
from tensorflow.python.framework.versions import COMPILER_VERSION as __compiler_version__
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as __cxx11_abi_flag__
from tensorflow.python.framework.versions import GIT_VERSION as __git_version__
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as __monolithic_build__
from tensorflow.python.framework.versions import VERSION as __version__
from tensorflow.python.module.module import Module
from tensorflow.python.ops.array_ops import batch_to_space_v2 as batch_to_space
from tensorflow.python.ops.array_ops import boolean_mask_v2 as boolean_mask
from tensorflow.python.ops.array_ops import broadcast_dynamic_shape
from tensorflow.python.ops.array_ops import broadcast_static_shape
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.array_ops import edit_distance
from tensorflow.python.ops.array_ops import expand_dims_v2 as expand_dims
from tensorflow.python.ops.array_ops import fill
from tensorflow.python.ops.array_ops import fingerprint
from tensorflow.python.ops.array_ops import gather_nd_v2 as gather_nd
from tensorflow.python.ops.array_ops import gather_v2 as gather
from tensorflow.python.ops.array_ops import identity
from tensorflow.python.ops.array_ops import meshgrid
from tensorflow.python.ops.array_ops import newaxis
from tensorflow.python.ops.array_ops import one_hot
from tensorflow.python.ops.array_ops import ones
from tensorflow.python.ops.array_ops import ones_like_v2 as ones_like
from tensorflow.python.ops.array_ops import pad_v2 as pad
from tensorflow.python.ops.array_ops import parallel_stack
from tensorflow.python.ops.array_ops import rank
from tensorflow.python.ops.array_ops import repeat
from tensorflow.python.ops.array_ops import required_space_to_batch_paddings
from tensorflow.python.ops.array_ops import reshape
from tensorflow.python.ops.array_ops import reverse_sequence_v2 as reverse_sequence
from tensorflow.python.ops.array_ops import searchsorted
from tensorflow.python.ops.array_ops import sequence_mask
from tensorflow.python.ops.array_ops import shape_n
from tensorflow.python.ops.array_ops import shape_v2 as shape
from tensorflow.python.ops.array_ops import size_v2 as size
from tensorflow.python.ops.array_ops import slice
from tensorflow.python.ops.array_ops import space_to_batch_v2 as space_to_batch
from tensorflow.python.ops.array_ops import split
from tensorflow.python.ops.array_ops import squeeze_v2 as squeeze
from tensorflow.python.ops.array_ops import stack
from tensorflow.python.ops.array_ops import strided_slice
from tensorflow.python.ops.array_ops import transpose_v2 as transpose
from tensorflow.python.ops.array_ops import unique
from tensorflow.python.ops.array_ops import unique_with_counts
from tensorflow.python.ops.array_ops import unstack
from tensorflow.python.ops.array_ops import where_v2 as where
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.array_ops import zeros_like_v2 as zeros_like
from tensorflow.python.ops.batch_ops import batch_function as nondifferentiable_batch_function
from tensorflow.python.ops.check_ops import assert_equal_v2 as assert_equal
from tensorflow.python.ops.check_ops import assert_greater_v2 as assert_greater
from tensorflow.python.ops.check_ops import assert_less_v2 as assert_less
from tensorflow.python.ops.check_ops import assert_rank_v2 as assert_rank
from tensorflow.python.ops.check_ops import ensure_shape
from tensorflow.python.ops.clip_ops import clip_by_global_norm
from tensorflow.python.ops.clip_ops import clip_by_norm
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.control_flow_ops import Assert
from tensorflow.python.ops.control_flow_ops import case_v2 as case
from tensorflow.python.ops.control_flow_ops import cond_for_tf_v2 as cond
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.control_flow_ops import switch_case
from tensorflow.python.ops.control_flow_ops import tuple_v2 as tuple
from tensorflow.python.ops.control_flow_ops import while_loop_v2 as while_loop
from tensorflow.python.ops.critical_section_ops import CriticalSection
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.ops.custom_gradient import grad_pass_through
from tensorflow.python.ops.custom_gradient import recompute_grad
from tensorflow.python.ops.functional_ops import foldl
from tensorflow.python.ops.functional_ops import foldr
from tensorflow.python.ops.functional_ops import scan
from tensorflow.python.ops.gen_array_ops import bitcast
from tensorflow.python.ops.gen_array_ops import broadcast_to
from tensorflow.python.ops.gen_array_ops import extract_volume_patches
from tensorflow.python.ops.gen_array_ops import guarantee_const
from tensorflow.python.ops.gen_array_ops import identity_n
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse
from tensorflow.python.ops.gen_array_ops import scatter_nd
from tensorflow.python.ops.gen_array_ops import space_to_batch_nd
from tensorflow.python.ops.gen_array_ops import stop_gradient
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add as tensor_scatter_nd_add
from tensorflow.python.ops.gen_array_ops import tensor_scatter_sub as tensor_scatter_nd_sub
from tensorflow.python.ops.gen_array_ops import tensor_scatter_update as tensor_scatter_nd_update
from tensorflow.python.ops.gen_array_ops import tile
from tensorflow.python.ops.gen_array_ops import unravel_index
from tensorflow.python.ops.gen_control_flow_ops import no_op
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition
from tensorflow.python.ops.gen_data_flow_ops import dynamic_stitch
from tensorflow.python.ops.gen_linalg_ops import matrix_square_root
from tensorflow.python.ops.gen_logging_ops import timestamp
from tensorflow.python.ops.gen_math_ops import acos
from tensorflow.python.ops.gen_math_ops import acosh
from tensorflow.python.ops.gen_math_ops import add
from tensorflow.python.ops.gen_math_ops import asin
from tensorflow.python.ops.gen_math_ops import asinh
from tensorflow.python.ops.gen_math_ops import atan
from tensorflow.python.ops.gen_math_ops import atan2
from tensorflow.python.ops.gen_math_ops import atanh
from tensorflow.python.ops.gen_math_ops import cos
from tensorflow.python.ops.gen_math_ops import cosh
from tensorflow.python.ops.gen_math_ops import exp
from tensorflow.python.ops.gen_math_ops import floor
from tensorflow.python.ops.gen_math_ops import greater
from tensorflow.python.ops.gen_math_ops import greater_equal
from tensorflow.python.ops.gen_math_ops import less
from tensorflow.python.ops.gen_math_ops import less_equal
from tensorflow.python.ops.gen_math_ops import lin_space as linspace
from tensorflow.python.ops.gen_math_ops import logical_and
from tensorflow.python.ops.gen_math_ops import logical_not
from tensorflow.python.ops.gen_math_ops import logical_or
from tensorflow.python.ops.gen_math_ops import maximum
from tensorflow.python.ops.gen_math_ops import minimum
from tensorflow.python.ops.gen_math_ops import neg as negative
from tensorflow.python.ops.gen_math_ops import real_div as realdiv
from tensorflow.python.ops.gen_math_ops import sign
from tensorflow.python.ops.gen_math_ops import sin
from tensorflow.python.ops.gen_math_ops import sinh
from tensorflow.python.ops.gen_math_ops import sqrt
from tensorflow.python.ops.gen_math_ops import square
from tensorflow.python.ops.gen_math_ops import tan
from tensorflow.python.ops.gen_math_ops import tanh
from tensorflow.python.ops.gen_math_ops import truncate_div as truncatediv
from tensorflow.python.ops.gen_math_ops import truncate_mod as truncatemod
from tensorflow.python.ops.gen_string_ops import as_string
from tensorflow.python.ops.gradients_impl import HessiansV2 as hessians
from tensorflow.python.ops.gradients_impl import gradients_v2 as gradients
from tensorflow.python.ops.gradients_util import AggregationMethod
from tensorflow.python.ops.histogram_ops import histogram_fixed_width
from tensorflow.python.ops.histogram_ops import histogram_fixed_width_bins
from tensorflow.python.ops.init_ops_v2 import Constant as constant_initializer
from tensorflow.python.ops.init_ops_v2 import Ones as ones_initializer
from tensorflow.python.ops.init_ops_v2 import RandomNormal as random_normal_initializer
from tensorflow.python.ops.init_ops_v2 import RandomUniform as random_uniform_initializer
from tensorflow.python.ops.init_ops_v2 import Zeros as zeros_initializer
from tensorflow.python.ops.linalg_ops import eye
from tensorflow.python.ops.linalg_ops import norm_v2 as norm
from tensorflow.python.ops.logging_ops import print_v2 as print
from tensorflow.python.ops.manip_ops import roll
from tensorflow.python.ops.map_fn import map_fn
from tensorflow.python.ops.math_ops import abs
from tensorflow.python.ops.math_ops import add_n
from tensorflow.python.ops.math_ops import argmax_v2 as argmax
from tensorflow.python.ops.math_ops import argmin_v2 as argmin
from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.math_ops import complex
from tensorflow.python.ops.math_ops import cumsum
from tensorflow.python.ops.math_ops import divide
from tensorflow.python.ops.math_ops import equal
from tensorflow.python.ops.math_ops import matmul
from tensorflow.python.ops.math_ops import multiply
from tensorflow.python.ops.math_ops import not_equal
from tensorflow.python.ops.math_ops import pow
from tensorflow.python.ops.math_ops import range
from tensorflow.python.ops.math_ops import reduce_all
from tensorflow.python.ops.math_ops import reduce_any
from tensorflow.python.ops.math_ops import reduce_logsumexp
from tensorflow.python.ops.math_ops import reduce_max
from tensorflow.python.ops.math_ops import reduce_mean
from tensorflow.python.ops.math_ops import reduce_min
from tensorflow.python.ops.math_ops import reduce_prod
from tensorflow.python.ops.math_ops import reduce_sum
from tensorflow.python.ops.math_ops import round
from tensorflow.python.ops.math_ops import saturate_cast
from tensorflow.python.ops.math_ops import scalar_mul_v2 as scalar_mul
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import subtract
from tensorflow.python.ops.math_ops import tensordot
from tensorflow.python.ops.math_ops import truediv
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.script_ops import eager_py_func as py_function
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.ops.sort_ops import argsort
from tensorflow.python.ops.sort_ops import sort
from tensorflow.python.ops.special_math_ops import einsum
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops.tensor_array_ops import TensorArraySpec
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.ops.variable_scope import variable_creator_scope
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.variables import VariableAggregationV2 as VariableAggregation
from tensorflow.python.ops.variables import VariableSynchronization
from tensorflow.python.platform.tf_logging import get_logger


from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v2", public_apis=None, deprecation=False,
      has_lite=False)


# Hook external TensorFlow modules.
_current_module = _sys.modules[__name__]
try:
  from tensorboard.summary._tf import summary
  _current_module.__path__ = (
      [_module_util.get_parent_dir(summary)] + _current_module.__path__)
  # Make sure we get the correct summary module with lazy loading
  setattr(_current_module, "summary", summary)
except ImportError:
  _logging.warning(
      "Limited tf.compat.v2.summary API due to missing TensorBoard "
      "installation.")

try:
  from tensorflow_estimator.python.estimator.api._v2 import estimator
  _current_module.__path__ = (
      [_module_util.get_parent_dir(estimator)] + _current_module.__path__)
  setattr(_current_module, "estimator", estimator)
except ImportError:
  pass

try:
  from tensorflow.python.keras.api._v2 import keras
  _current_module.__path__ = (
      [_module_util.get_parent_dir(keras)] + _current_module.__path__)
  setattr(_current_module, "keras", keras)
except ImportError:
  pass


# We would like the following to work for fully enabling 2.0 in a 1.0 install:
#
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
#
# This make this one symbol available directly.
from tensorflow.python.compat.v2_compat import enable_v2_behavior  # pylint: disable=g-import-not-at-top
setattr(_current_module, "enable_v2_behavior", enable_v2_behavior)

# Add module aliases
if hasattr(_current_module, 'keras'):
  losses = keras.losses
  metrics = keras.metrics
  optimizers = keras.optimizers
  initializers = keras.initializers
  setattr(_current_module, "losses", losses)
  setattr(_current_module, "metrics", metrics)
  setattr(_current_module, "optimizers", optimizers)
  setattr(_current_module, "initializers", initializers)
