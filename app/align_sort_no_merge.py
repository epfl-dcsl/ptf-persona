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
import itertools
from . import app
from modules.snap_align import fused_align_sort, merge as merge_stage
from common.parse import numeric_min_checker, add_dataset
import tensorflow.contrib.gate as gate
import tensorflow as tf
import logging; logging.basicConfig(level=logging.DEBUG)
import tensorflow.contrib.persona as persona
persona_ops = persona.persona_ops()
from .common import make_counter

align_sort_key = "align_sort"
rm_key = "rm"

performance_name_scope = "performance"

credit_link_end_to_end = "e2e"
credit_link_successive = "linear"

def add_common_args(parser):
    parser.add_argument("--align-stages", dest="align_stages", default=0, type=numeric_min_checker(0, "must have at least 1 align fused_align_sort"), help="number of align stages")
    parser.add_argument("--rm-stages", dest="rm_stages", default=1, type=numeric_min_checker(0, "must have at least 1 rm stage"), help="number of merge stages")
    parser.add_argument("--parallel-open-requests", type=numeric_min_checker(1, "must have at least 1 parallel open request"), help="if specified, the number of parallel open requests")
    parser.add_argument("--parallel-open-request-expansion-factor", default=1.5, type=numeric_min_checker(0.1, numeric_type=float, message="must have at least 0.1 expansion factor"),
                        help="the expansion factor to multiple the number of client slots by to bound the capacity in the global pipeline. Not used if parallel_open_requests is set")
    parser.add_argument("--credit-link", default=credit_link_successive, choices=(credit_link_end_to_end, credit_link_successive), help="Type of credit linking to use between successive stages")
    parser.add_argument("--align-counters", default=False, action="store_true", help="track the exit rate of the align/sort stages")

class NullCephAlignSort(app.Application):

    ingress_dtypes = (tf.string,)
    ingress_shapes = ((2),)

    @staticmethod
    def name():
        return "null-ceph-align-sort"

    @staticmethod
    def help_message():
        return "align a dataset with Ceph but don't merge"

    class_logger = logging.getLogger(name="NullCephAlignClass")

    @classmethod
    def _make_graph_args(cls, parser):
        add_common_args(parser=parser)
        parser.add_argument("--log-goodput", default=False, action='store_true', help="turn on all goodput and latency tracing")
        fused_align_sort.CephFusedStage.add_graph_args(parser=parser)
        merge_stage.NullMergeStage.add_graph_args(parser=parser)

    @classmethod
    def device_counts(cls, args):
        return {
            align_sort_key: args.align_stages,
            rm_key: args.rm_stages,
        }

    def _construct_graph(self, args, device_map, num_client_slots):
        gate_name = "ingress_gate"

        num_rm = args.rm_stages
        num_align = args.align_stages

        if args.parallel_open_requests is not None:
            capacity_between_gates = args.parallel_open_requests
        else:
            capacity_between_gates = int(num_client_slots * args.parallel_open_request_expansion_factor)
        if capacity_between_gates < 1:
            raise Exception("Capacity between gates is <1 ({c})".format(c=capacity_between_gates))
        args.parallel_open_requests = capacity_between_gates
        self.log.info("Capacity between gates: {}".format(capacity_between_gates))

        with tf.name_scope(gate_name):
            ingress = gate.IngressGate(dtypes=self.ingress_dtypes, shapes=self.ingress_shapes, capacity=capacity_between_gates,
                                       shared_name=gate_name, name=gate_name)

        with tf.name_scope("align_sort_stage"):
            align_stages = tuple(fused_align_sort.CephFusedStage(args=args) for _ in range(num_align))
            def make_align_stages(stages, align_devices):
                for stage, device in zip(stages, align_devices):
                    with device():
                        device_graph = stage.make_graph(upstream_gate=ingress)
                        try: # convert to a tuple if it returns a generator
                            device_graph[0]
                        except TypeError:
                            device_graph = tuple(device_graph)
                        assert len(stage.run_first) > 0
                        for item in stage.run_first:
                            self._add_run_first(tensor=item)
                        yield device_graph
            outputs = tuple(itertools.chain.from_iterable(
                make_align_stages(stages=s, align_devices=devices) for s, devices in (
                    (align_stages, device_map.get(align_sort_key, None)),
                ) if devices is not None
            ))
        assert len(outputs) == num_align, "Expected {e} align stage, but only got {actual}".format(
            e=num_align, actual=len(outputs))

        outputs = tuple(itertools.chain.from_iterable(outputs)) # flattens it
        example_output = outputs[0]
        if args.credit_link == credit_link_end_to_end:
            merge_gate_kwargs = {
                "limit_upstream": False,
                "limit_downstream": False
            }
        else:
            merge_gate_kwargs = {
                "capacity": capacity_between_gates
            }
        with tf.name_scope("inter_stage_gate"):
            gate_name = "ready_to_merge"
            merge_gate = gate.StreamingGate(
                sample_tensors=example_output[1:-1],
                id_and_count_upstream=example_output[0], join=True,
                name=gate_name, shared_name=gate_name,
                **merge_gate_kwargs
            )
            enqueue_ops = tuple(merge_gate.enqueue(id_and_count=a[0], components=a[1:-1]) for a in outputs)
            if args.align_counters:
                if getattr(args, "summary", False):
                    with tf.name_scope(None):
                        with tf.name_scope(performance_name_scope):
                            enqueue_ops = tuple(make_counter(counter_name="sorted_counter",
                                                             summary_name="sorted_num_records",
                                                             deps_and_counters=zip(
                                                                 enqueue_ops,
                                                                 (a[-1] for a in outputs)
                                                             )))
                else:
                    self.log.warning("Align counters requested, but no summary was requested. Please enable summary for this to work")
            gate.add_gate_runner(gate_runner=gate.GateRunner(gate=merge_gate, enqueue_ops=enqueue_ops, device=merge_gate.device))
            if args.credit_link == credit_link_successive:
                gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=merge_gate)

        with tf.name_scope("rm_stage"):
            merge_stages = tuple(merge_stage.NullMergeStage(args=args) for _ in range(num_rm))

            def make_merge_stages(stages, merge_devices):
                for stage, device in zip(stages, merge_devices):
                    with device():
                        device_graph = stage.make_graph(upstream_gate=merge_gate)
                        try:
                            device_graph[0]
                        except TypeError:
                            device_graph = tuple(device_graph)
                        yield device_graph

            merge_stage_outputs = tuple(itertools.chain.from_iterable(
                make_merge_stages(stages=s, merge_devices=devices) for s, devices in (
                    (merge_stages, device_map.get(rm_key, None)),
                ) if devices is not None
            ))
        assert len(merge_stage_outputs) == num_rm, "Expected {e} merge devices, but only got {actual}".format(
            e=num_rm, actual=len(merge_stage_outputs)
        )

        merge_stage_outputs = tuple(itertools.chain.from_iterable(merge_stage_outputs)) # flattens it
        example_output = merge_stage_outputs[0]
        gate_name = "egress_gate"
        with tf.name_scope(gate_name):
            egress = gate.EgressGate(capacity=capacity_between_gates, sample_tensors=example_output[1:],
                                     id_and_count_upstream=example_output[0],
                                     name=gate_name, shared_name=gate_name)
            enqueue_ops = tuple(egress.enqueue(id_and_count=a[0], components=a[1:]) for a in merge_stage_outputs)
            gate.add_gate_runner(gate_runner=gate.GateRunner(gate=egress, enqueue_ops=enqueue_ops,
                                                             device=egress.device))
            if args.credit_link == credit_link_end_to_end:
                gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=egress)
            else:
                gate.add_credit_supplier_from_gates(upstream_gate=merge_gate, downstream_gate=egress)

        self.close_op = (ingress.close(), egress.close())

        with tf.name_scope("client_slots"):
            unknown_shape = tf.TensorShape([None])
            batch_ingress_shapes = tuple(unknown_shape.concatenate(ishape) for ishape in self.ingress_shapes)
            for idx in range(num_client_slots):
                ingress_placeholders = tuple(tf.placeholder(dtype=dtype, shape=shape, name="client_slot_{}".format(idx)) for dtype, shape in zip(self.ingress_dtypes, batch_ingress_shapes))
                ingress_enqueue = ingress.enqueue_request(components=ingress_placeholders, name="ingress_enqueue_{}".format(idx))
                egress_dequeue = egress.dequeue_request(request_id=ingress_enqueue, name="egress_dequeue_{}".format(idx))
                yield self.ClientSlot(ingress_placeholders=ingress_placeholders, egress_dequeue=egress_dequeue)

    @classmethod
    def make_client_args(cls, parser):
        add_dataset(parser=parser)
        parser.add_argument("--namespace", default="", help="the namespace to access this dataset")
        parser.add_argument("--use-default-namespace", default=False, action="store_true", help="use the name of this record as the namespace")

    @classmethod
    def process_ingress_args(cls, args):
        dataset = args.dataset
        if args.use_default_namespace:
            namespace = dataset["name"]
        else:
            namespace = args.namespace
        if namespace == "":
            cls.class_logger.warning("Dataset {rid} has no namespace specified!".format(rid=dataset["name"]))
        record_keys = (a["path"] for a in dataset["records"])
        return tuple(zip(record_keys, itertools.repeat(namespace)))

    @classmethod
    def process_egress_results(cls, results, args):
        """
        :param results: a list of [ [ names, of, intermediate, files] ]
        :param args:
        :return:
        """
        pass

    def _run_client_request(self, client_args, client_slot, sess):
        client_args = tuple(client_args)
        ingress_placeholder = client_slot.ingress_placeholders[0]
        egress_dequeue = client_slot.egress_dequeue
        sess.run(egress_dequeue, feed_dict={ingress_placeholder: client_args})

    def stop(self, sess):
        try:
            sess.run(self.close_op)
        except Exception as e:
            self.log.error("{nm} closing. Got exception '{e}'".format(e=e, nm=self.name()))
