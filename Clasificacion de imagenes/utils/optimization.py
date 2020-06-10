# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      power=1.0,
      name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(global_step_float < warmup_steps_float,
                     lambda: warmup_learning_rate,
                     lambda: self.decay_schedule_fn(step),
                     name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }

def warmupdecay_learner(batch_size=64,                 
                     epoch_size=10000,
                     init_lr=0.01,
                     warmup_epochs=5,
                     boundaries=[30, 60, 80],
                     multipliers=[1.0, 0.1, 0.01, 0.001]):
    
  steps_per_epoch = epoch_size // batch_size
  warmup_steps = warmup_epochs * steps_per_epoch
  step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
  lr_values = [init_lr * m for m in multipliers]

  learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries=step_boundaries,
      values=lr_values)

  if warmup_steps>0:
    learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                          decay_schedule_fn=learning_rate_fn,
                          warmup_steps=warmup_steps)

  return learning_rate_fn


def expdecay_learner(batch_size=64,                 
                     epoch_size=10000,
                     init_lr=0.01,
                     decay_epochs=5,
                     decay_rate=0.97):
    
  steps_per_epoch = epoch_size // batch_size
  decay_steps = int(decay_epochs * steps_per_epoch)

  learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=init_lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate)

  return learning_rate_fn