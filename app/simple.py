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
from app import app
from modules.simple import stage as simple_stage
from common.parse import numeric_min_checker
import itertools
import logging
logging.basicConfig(level=logging.DEBUG)
import tensorflow as tf
import tensorflow.contrib.gate as gate
class Simple(app.Application):

    app_dtypes = (tf.int32,)
    app_shapes = ((),)

    log = logging.getLogger("Simple")
    log.setLevel(level=logging.DEBUG) # don't know why basicConfig() isn't doing this

    @staticmethod
    def name():
        return "simple"

    @staticmethod
    def help_message():
        return "run a simple increment app"

    @classmethod
    def device_counts(cls, args):
        return { "": args.stages }

    @classmethod
    def _make_graph_args(cls, parser):
        simple_stage.Incrementer.add_graph_args(parser=parser)
        parser.add_argument("--stages", default=1, type=numeric_min_checker(minimum=1, message="need at least one stage!"), help="number of stages to run in parallel")

    def _construct_graph(self, args, device_map, num_client_slots):
        gate_name = "ingress_gate"
        capacity_between_gates = int(num_client_slots*1.5)
        ingress = gate.IngressGate(dtypes=self.app_dtypes, shapes=self.app_shapes, capacity=capacity_between_gates,
                                   shared_name=gate_name, name=gate_name)

        stages = tuple(simple_stage.Incrementer(args=args) for _ in range(args.stages))
        devices = device_map[""]
        assert len(devices) == len(stages)
        def make_outputs():
            for device, stage in zip(devices, stages):
                with device:
                   yield stage.make_graph(upstream_gate=ingress)

        outputs = tuple(make_outputs())
        example_output = outputs[0]
        egress = gate.EgressGate(capacity=capacity_between_gates, sample_tensors=example_output[1:], id_and_count_upstream=example_output[0], join=True,
                                 name="egress_gate")
        enqueue_ops = tuple(egress.enqueue(id_and_count=a[0], components=a[1:]) for a in outputs)
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=egress, enqueue_ops=enqueue_ops,
                                                         device=egress.device)) # ideally, each local device would run its own gate runner, but we're running everything locally to make it easy
        gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=egress)
        self.close_op = (ingress.close(), egress.close())

        unknown_shape = tf.TensorShape([None])
        batch_ingress_shapes = tuple(unknown_shape.concatenate(ishape) for ishape in self.app_shapes)
        for _ in range(num_client_slots):
            ingress_placeholders = tuple(tf.placeholder(dtype=dtype, shape=shape) for dtype, shape in zip(self.app_dtypes, batch_ingress_shapes))
            ingress_enqueue = ingress.enqueue_request(components=ingress_placeholders)
            egress_dequeue = egress.dequeue_request(request_id=ingress_enqueue)
            yield self.ClientSlot(ingress_placeholders=ingress_placeholders, egress_dequeue=egress_dequeue)

    @classmethod
    def make_client_args(cls, parser):
        parser.add_argument("numbers", nargs="+", type=int, help="integers to increment in this pipeline")

    @classmethod
    def process_ingress_args(cls, args):
        return args.numbers

    @classmethod
    def process_egress_results(cls, results, args):
        cls.log.info("Got results: {}".format(results))

    def _run_client_request(self, client_args, client_slot, sess):
        client_args = tuple(client_args)
        ingress_placeholder = client_slot.ingress_placeholders[0]
        egress_dequeue = client_slot.egress_dequeue
        a = sess.run(egress_dequeue, feed_dict={ingress_placeholder: client_args})
        return tuple(itertools.chain.from_iterable(a[0].tolist())) #flattens it out to (0,1,2) instead of ((0),(1),(2))

    def stop(self, sess):
        sess.run(self.close_op)
