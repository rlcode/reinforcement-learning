# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ===================================================================
"""Tooling for support TPU embedding in TPUEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow.python.training import training
from tensorflow.python.util.tf_export import estimator_export
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib

# pylint: disable=protected-access
_TPU_EMBEDDING_COLUMN_CLASSES = (tpu_fc._TPUEmbeddingColumn,
                                 tpu_fc._TPUSharedEmbeddingColumn,
                                 tpu_fc_v2._TPUEmbeddingColumnV2,
                                 tpu_fc_v2._TPUSharedEmbeddingColumnV2)
_EMBEDDING_COLUMN_CLASSES = (core_fc._EmbeddingColumn,
                             core_fc_lib.EmbeddingColumn,
                             core_fc._SharedEmbeddingColumn)
_SUPPORTED_FEATURE_COLUMNS = (core_fc._NumericColumn, core_fc_lib.NumericColumn)
_SUPPORTED_OPTIMIZERS = (AdagradParameters, AdamParameters,
                         StochasticGradientDescentParameters)

# pylint: enable=protected-access

_TABLE_NAME_PREFIX = 'tbl_'
_LEN_TABLE_NAME_PREFIX = len(_TABLE_NAME_PREFIX)


def _get_table_name_from_embedding_var_name(embedding_var_name):
  return '{}{}'.format(_TABLE_NAME_PREFIX, embedding_var_name)


def _get_embedding_var_name_from_table_name(table_name):
  return table_name[_LEN_TABLE_NAME_PREFIX:]


def _get_embedding_variable_name(scope_name, var_name):
  if scope_name:
    scope_name = scope_name + '/'
  return '{}{}'.format(scope_name, var_name)


def _get_slot_variable_names(scope_name, var_name, optimization_parameters):
  """Return embedding variable names which are consistent with CPU runs."""
  if scope_name:
    scope_name = scope_name + '/'
  if isinstance(optimization_parameters, tpu_embedding.AdagradParameters):
    return tpu_embedding.AdagradSlotVariableName(
        '{}{}/Adagrad'.format(scope_name, var_name)
    )
  elif isinstance(optimization_parameters, tpu_embedding.AdamParameters):
    return tpu_embedding.AdamSlotVariableNames(
        '{}{}/Adam/m'.format(scope_name, var_name),
        '{}{}/Adam/v'.format(scope_name, var_name)
    )
  elif isinstance(optimization_parameters,
                  tpu_embedding.StochasticGradientDescentParameters):
    return None
  else:
    raise ValueError('Support to infer full variable name '
                     'for optimization_parameter {} has not been added.'
                     .format(optimization_parameters))


def get_full_variable_names(
    graph, table_to_config_dict, optimization_parameters=None):
  """Return embedding variable names and slot variables which are consistent with CPU runs."""
  collection = graph.get_collection_ref(tpu_fc._TPU_FC_TO_SCOPE)  # pylint: disable=protected-access
  if not collection:
    raise RuntimeError(
        'Embedding feature column did not capture any thing. Make sure the '
        'feature columns passed to TPUEstimator constructor is properly '
        'used in model_fn.')

  embedding_variable_name_by_table = {}
  slot_variable_names_by_table = {}
  for table_name in table_to_config_dict:
    embedding_var_name = _get_embedding_var_name_from_table_name(table_name)
    (scope_name, var_name) = collection[0][embedding_var_name]
    embedding_variable_name_by_table[table_name] = (
        _get_embedding_variable_name(scope_name, var_name))
    if optimization_parameters:
      slot_variable_names_by_table[table_name] = _get_slot_variable_names(
          scope_name, var_name, optimization_parameters)

  graph.clear_collection(tpu_fc._TPU_FC_TO_SCOPE)  # pylint: disable=protected-access
  return embedding_variable_name_by_table, slot_variable_names_by_table


def get_lr(feature_columns):
  lrs = list({col.get_learning_rate_fn() for col in feature_columns
              if col.get_learning_rate_fn() is not None})
  lr_key_to_fn = {str(i): lr for i, lr in enumerate(lrs)}
  lr_fn_to_key = {v: k for k, v in six.iteritems(lr_key_to_fn)}
  return lr_key_to_fn, lr_fn_to_key


def get_configs_from_feature_columns(feature_columns):
  """Create configs for TPUEmbedding etc from a list of feature columns.

  Args:
    feature_columns: a list of supported feature columns.

  Returns:
    A tuple of dicts, the first maps tables to their config, the second maps
    features to their config, the third maps learning rate key to callback that
    takes global step and outputs dynamic learning rate.
  """

  allowed = (tpu_fc_v2._TPUEmbeddingColumnV2,  # pylint: disable=protected-access
             tpu_fc_v2._TPUSharedEmbeddingColumnV2)  # pylint: disable=protected-access
  warn = (tpu_fc._TPUEmbeddingColumn, tpu_fc._TPUSharedEmbeddingColumn)  # pylint: disable=protected-access

  for column in feature_columns:
    if not isinstance(column, allowed + warn):
      raise TypeError(
          'Unsupported feature column {}. Supported types are {}.'.format(
              type(column), allowed))
    if isinstance(column, warn):
      logging.warn(
          'Columns of type {} are deprecated. Supported types are {}.'.format(
              type(column), allowed))

  table_to_config = {}
  feature_to_config = {}
  lr_key_to_fn, lr_fn_to_key = get_lr(feature_columns)
  lr_fn_to_key[None] = None
  for column in feature_columns:
    feature_name = column.get_feature_key_name()
    table_name = _get_table_name_from_embedding_var_name(
        column.get_embedding_var_name())
    if feature_name in feature_to_config:
      raise ValueError(
          'Feature column {} is used with multiple embeddings and this is '
          'not supported.'.format(feature_name))
    feature_to_config[feature_name] = tpu_embedding.FeatureConfig(
        table_id=table_name,
        max_sequence_length=column.get_max_sequence_length(),
        weight_key=column.get_weight_key_name())
    vocabulary_size, dimension = column.get_embedding_table_size()
    table_to_config[table_name] = tpu_embedding.TableConfig(
        vocabulary_size=vocabulary_size,
        dimension=dimension,
        initializer=column.get_initializer(),
        combiner=column.get_combiner(),
        learning_rate_key=lr_fn_to_key[column.get_learning_rate_fn()])

  return table_to_config, feature_to_config, lr_key_to_fn


@estimator_export(v1=['estimator.tpu.experimental.EmbeddingConfigSpec'])
class EmbeddingConfigSpec(
    collections.namedtuple('EmbeddingConfigSpec', [
        'feature_columns', 'optimization_parameters', 'clipping_limit',
        'pipeline_execution_with_tensor_core',
        'experimental_gradient_multiplier_fn',
        'feature_to_config_dict', 'table_to_config_dict', 'partition_strategy'
    ])):
  """Class to keep track of the specification for TPU embeddings.

  Pass this class to `tf.estimator.tpu.TPUEstimator` via the
  `embedding_config_spec` parameter. At minimum you need to specify
  `feature_columns` and `optimization_parameters`. The feature columns passed
  should be created with some combination of
  `tf.tpu.experimental.embedding_column` and
  `tf.tpu.experimental.shared_embedding_columns`.

  TPU embeddings do not support arbitrary Tensorflow optimizers and the
  main optimizer you use for your model will be ignored for the embedding table
  variables. Instead TPU embeddigns support a fixed set of predefined optimizers
  that you can select from and set the parameters of. These include adagrad,
  adam and stochastic gradient descent. Each supported optimizer has a
  `Parameters` class in the `tf.tpu.experimental` namespace.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  column_c = tf.feature_column.categorical_column_with_identity(...)
  tpu_shared_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  tpu_non_shared_column = tf.tpu.experimental.embedding_column(
      column_c, 10)
  tpu_columns = [tpu_non_shared_column] + tpu_shared_columns
  ...
  def model_fn(features):
    dense_features = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_features(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          optimization_parameters=(
              tf.estimator.tpu.experimental.AdagradParameters(0.1))))
  """

  def __new__(cls,
              feature_columns=None,
              optimization_parameters=None,
              clipping_limit=None,
              pipeline_execution_with_tensor_core=False,
              experimental_gradient_multiplier_fn=None,
              feature_to_config_dict=None,
              table_to_config_dict=None,
              partition_strategy='div'):
    """Creates an `EmbeddingConfigSpec` instance.

    Args:
      feature_columns: All embedding `FeatureColumn`s used by model.
      optimization_parameters: An instance of `AdagradParameters`,
        `AdamParameters` or `StochasticGradientDescentParameters`. This
        optimizer will be applied to all embedding variables specified by
        `feature_columns`.
      clipping_limit: (Optional) Clipping limit (absolute value).
      pipeline_execution_with_tensor_core: setting this to `True` makes training
        faster, but trained model will be different if step N and step N+1
        involve the same set of embedding IDs. Please see
        `tpu_embedding_configuration.proto` for details.
      experimental_gradient_multiplier_fn: (Optional) A Fn taking global step as
        input returning the current multiplier for all embedding gradients.
      feature_to_config_dict: A dictionary mapping features names to instances
        of the class `FeatureConfig`. Either features_columns or the pair of
        `feature_to_config_dict` and `table_to_config_dict` must be specified.
      table_to_config_dict: A dictionary mapping features names to instances of
        the class `TableConfig`. Either features_columns or the pair of
        `feature_to_config_dict` and `table_to_config_dict` must be specified.
      partition_strategy: A string, determining how tensors are sharded to the
        tpu hosts. See `tf.nn.safe_embedding_lookup_sparse` for more details.
        Allowed value are `"div"` and `"mod"'. If `"mod"` is used, evaluation
        and exporting the model to CPU will not work as expected.

    Returns:
      An `EmbeddingConfigSpec` instance.

    Raises:
      ValueError: If the feature_columns are not specified.
      TypeError: If the feature columns are not of ths correct type (one of
        _SUPPORTED_FEATURE_COLUMNS, _TPU_EMBEDDING_COLUMN_CLASSES OR
        _EMBEDDING_COLUMN_CLASSES).
      ValueError: If `optimization_parameters` is not one of the required types.
    """
    if (not feature_columns and not (feature_to_config_dict and
                                     table_to_config_dict)
        or (feature_columns and (feature_to_config_dict
                                 and table_to_config_dict))):
      raise ValueError('Exactly one of `feature_columns` and the pair '
                       '`feature_to_config_dict` and `table_to_config_dict` '
                       'must be be specified.')

    if partition_strategy not in ('div', 'mod'):
      raise ValueError('Invalid partition_strategy {}. Must be one of "mod" or '
                       '"div".'.format(partition_strategy))

    if feature_columns:
      # It is unknown at this moment, whether the TPUEstimator is running in CPU
      # or TPU mode. So allow non-TPU embedding columns also.
      supported_classes = tuple(
          list(_SUPPORTED_FEATURE_COLUMNS) +
          list(_TPU_EMBEDDING_COLUMN_CLASSES) +
          list(_EMBEDDING_COLUMN_CLASSES))

      for column in feature_columns:
        if not isinstance(column, supported_classes):
          raise TypeError(
              'All feature columns must be supported types in {}. Got {}'
              .format(supported_classes, type(column)))

      if not isinstance(optimization_parameters, _SUPPORTED_OPTIMIZERS):
        raise ValueError('optimization_parameters must be an instance of type '
                         '{}. Got {}.'.format(_SUPPORTED_OPTIMIZERS,
                                              type(optimization_parameters)))
    else:
      for feature, config in feature_to_config_dict.items():
        if not isinstance(config, tpu_embedding.FeatureConfig):
          raise TypeError(
              'Config for feature {} must be of type `FeatureConfig`. Got {}'
              .format(feature, type(config)))
        if config.table_id not in table_to_config_dict:
          raise ValueError('Feature {} refers to table {} which is not in the '
                           'table_to_config_dict.'.format(feature,
                                                          config.table_id))
      for table, config in table_to_config_dict.items():
        if not isinstance(config, tpu_embedding.TableConfig):
          raise TypeError(
              'Config for table {} must be of type `TableConfig`. Got '
              '{}'.format(table, type(config)))

    return super(EmbeddingConfigSpec, cls).__new__(
        cls,
        feature_columns=feature_columns,
        optimization_parameters=optimization_parameters,
        clipping_limit=clipping_limit,
        pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core,
        experimental_gradient_multiplier_fn=experimental_gradient_multiplier_fn,
        feature_to_config_dict=feature_to_config_dict,
        table_to_config_dict=table_to_config_dict,
        partition_strategy=partition_strategy)


class EmbeddingConfig(object):
  """This is the internal immutable object for embedding config.

  `_EmbeddingConfig` is responsible to _translate_ user provided
  `EmbeddingConfigSpec` to internal data structures, mostly constructor
  arguments of `TPUEmbedding`.
  """

  def __init__(self, embedding_config_spec, train_batch_size, eval_batch_size,
               num_hosts, num_cores, run_config):
    if not embedding_config_spec:
      raise ValueError('embedding_config_spec cannot be None.')

    self._embedding_config_spec = embedding_config_spec
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._num_hosts = num_hosts
    self._num_cores = num_cores
    self._run_config = run_config

    if embedding_config_spec.feature_columns:
      (self._table_to_config_dict, self._feature_to_config_dict,
       self._lr_key_to_fn) = (
           get_configs_from_feature_columns(
               embedding_config_spec.feature_columns))
    else:
      self._table_to_config_dict = embedding_config_spec.table_to_config_dict
      self._feature_to_config_dict = embedding_config_spec.feature_to_config_dict
      self._lr_key_to_fn = {}
    self._partition_strategy = embedding_config_spec.partition_strategy
    self._mode_to_tpu_embedding_dict = {}
    self.dummy_table_variables = None

    self._grad_multiplier_fn = (
        embedding_config_spec.experimental_gradient_multiplier_fn)

  def get_grad_multiplier(self):
    if self._grad_multiplier_fn:
      return ops.convert_to_tensor(
          self._grad_multiplier_fn(training.get_global_step()),
          dtype=dtypes.float32)

  def get_dynamic_learning_rates(self):
    return {k: math_ops.cast(
        fn(training.get_global_step()), dtype=dtypes.float32)
            for k, fn in six.iteritems(self._lr_key_to_fn)}

  def has_embedding_tables(self):
    return bool(self._table_to_config_dict)

  def _create_tpu_embedding(self, mode):
    """Create tpu_embedding.TPUEmbedding based on mode."""
    if mode == model_fn_lib.ModeKeys.TRAIN:
      batch_size = self._train_batch_size
    else:
      batch_size = self._eval_batch_size

    if mode == model_fn_lib.ModeKeys.TRAIN:
      tpu_embedding_mode = tpu_embedding.TRAINING
      optimization_parameters = (
          self._embedding_config_spec.optimization_parameters)
    elif (mode == model_fn_lib.ModeKeys.EVAL or
          mode == model_fn_lib.ModeKeys.PREDICT):
      tpu_embedding_mode = tpu_embedding.INFERENCE
      optimization_parameters = None
    else:
      raise ValueError('Mode {} is not supported.'.format(mode))

    if self._run_config.cluster:
      master = self._run_config.cluster.master()
      cluster_spec = self._run_config.cluster.cluster_spec()
      cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
    else:
      master = (
          self._run_config.evaluation_master
          if mode == model_fn_lib.ModeKeys.EVAL else self._run_config.master)
      cluster_def = None
    tpu_embedding_ = tpu_embedding.TPUEmbedding(
        self._table_to_config_dict,
        self._feature_to_config_dict,
        batch_size,
        tpu_embedding_mode,
        master,
        optimization_parameters,
        cluster_def,
        pipeline_execution_with_tensor_core=self._embedding_config_spec
        .pipeline_execution_with_tensor_core,
        partition_strategy=self._partition_strategy)
    return tpu_embedding_

  def get_tpu_embedding(self, mode):
    if mode not in self._mode_to_tpu_embedding_dict:
      self._mode_to_tpu_embedding_dict[mode] = (
          self._create_tpu_embedding(mode))
    return self._mode_to_tpu_embedding_dict[mode]


def split_inputs(ctx, features, labels):
  """Splits the dense and sparse tensors inside the features and labels."""
  enqueue_datas = collections.OrderedDict()
  if ctx.embedding_config:
    tpu_embedding_ = ctx.embedding_config.tpu_embedding
    for feature_key in tpu_embedding_.feature_to_config_dict:
      sparse_feature = _get_sparse_feature_from_feature(feature_key, features)
      max_sequence_length = tpu_embedding_.feature_to_config_dict[
          feature_key].max_sequence_length
      if max_sequence_length > 0:
        length_feature_name = (
            tpu_fc.get_sequence_length_feature_key_name_from_feature_key_name(
                feature_key))
        length_feature = math_ops.minimum(
            fc_utils.sequence_length_from_sparse_tensor(sparse_feature),
            max_sequence_length)
        length_feature.set_shape(ctx.batch_size_for_input_fn)
        features[length_feature_name] = length_feature
      weight_key = tpu_embedding_.feature_to_config_dict[feature_key].weight_key
      if isinstance(sparse_feature, sparse_tensor.SparseTensor):
        weights = _get_weights_from_features(weight_key, features)
        enqueue_data = tpu_embedding.EnqueueData.from_sparse_tensor(
            sparse_feature, weights)
      else:
        if weight_key is not None:
          raise ValueError(
              'Found weights {} for weighted_categorical_column, which is not'
              'compatible with sparse feature {} enqueued as dense tensor.'
              .format(weight_key, feature_key))
        enqueue_data = tpu_embedding.EnqueueData(sparse_feature)
      enqueue_datas[feature_key] = enqueue_data

  return features, labels, enqueue_datas


def _get_sparse_feature_from_feature(feature_key, features):
  """Pop and return sparse feature."""
  sparse_feature = features.pop(feature_key)
  if not sparse_feature.dtype.is_integer:
    raise ValueError('SparseTensor with string as values are not supported. '
                     'If you are using vocabulary_file_categorical_column or '
                     'vocabulary_list_categorical_column, please call '
                     'your_column.categorical_column._transform_feature({{'
                     'your_column.key: features[your_column.key]}}) in'
                     'your input_fn() to convert string to int. '
                     'feature_key = {}.'.format(feature_key))
  return sparse_feature


def _get_weights_from_features(weight_key_name, features):
  """Pop and return feature for weights, possibly None."""
  weights = None
  if weight_key_name is not None:
    if weight_key_name in features:
      weights = features.pop(weight_key_name)
    else:
      raise ValueError(
          'Cannot find weights {} for weighted_categorical_column.'
          ' Please check if the weights are present in feature dict. Also'
          ' note weight-sharing among weighted_categorical_column is not '
          'supported on TPU.'.format(weight_key_name))
    if not isinstance(weights, sparse_tensor.SparseTensor):
      raise ValueError(
          'weighted_categorical_column with weight key name {} has dense '
          'weights. Dense weights are not supported on TPU. Please use '
          'sparse weights instead.'.format(weight_key_name))
    if weights.dtype is not dtypes.float32:
      weights = math_ops.to_float(weights)
  return weights


def get_tpu_embedding_columns(feature_columns):
  """Get feature columns meant to use TPU embedding.

  Args:
    feature_columns: a list of feature columns.

  Returns:
    A list of feature columns which can be placed on TPU embedding.
  """
  tpu_embedding_columns = []
  for column in feature_columns:
    if isinstance(column, _TPU_EMBEDDING_COLUMN_CLASSES):
      tpu_embedding_columns.append(column)
  return tpu_embedding_columns
