# Copyright 2019 École Polytechnique Fédérale de Lausanne. All Rights Reserved.
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
import tensorflow as tf
import tensorflow.contrib.persona as persona
persona_ops = persona.persona_ops()

def make_counter(deps_and_counters, counter_name, summary_name):
    counter = persona_ops.atomic_counter(name=counter_name,
                                         shared_name=counter_name)
    for op_deps, counter_value in deps_and_counters:
        counter_value = tf.to_int64(counter_value)
        if not isinstance(op_deps, (list, tuple)):
            op_deps =(op_deps,)
        with tf.control_dependencies(op_deps):
            incr_op = persona_ops.atomic_counter_incrementer(
                counter=counter,
                delta=counter_value
            )
            yield incr_op
    tf.summary.scalar(
        name=summary_name,
        tensor=persona_ops.atomic_counter_fetch_and_set(
            counter=counter,
            new_value=0
        )
    )
