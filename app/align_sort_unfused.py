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
import pathlib
from . import app
from modules.snap_align import snap_align, merge as merge_stage, sort as sort_stage, common as snap_common
from common.parse import numeric_min_checker, add_dataset, path_exists_checker, filepath_key
import tensorflow.contrib.gate as gate
import tensorflow as tf
import logging; logging.basicConfig(level=logging.DEBUG)
import tensorflow.contrib.persona as persona
persona_ops = persona.persona_ops()
from string import digits
import json
from .common import make_counter

align_key = "align"
sort_key = "sort"
merge_key = "merge"

performance_name_scope = "performance"

credit_link_end_to_end = "e2e"
credit_link_successive = "linear"

def add_common_args(parser):
    parser.add_argument("--align-stages", dest="align_stages", type=numeric_min_checker(1, "must have at least 1 align stage"), help="number of align stages")
    parser.add_argument("--sort-stages", dest="sort_stages", type=numeric_min_checker(1, "must have at least 1 sort stage"), help="number of sort stages")
    parser.add_argument("--merge-stages", dest="merge_stages", type=numeric_min_checker(1, "must have at least 1 merge stage"), help="number of merge stages")
    parser.add_argument("--parallel-open-requests", type=numeric_min_checker(1, "must have at least 1 parallel open request"), help="if specified, the number of parallel open requests")
    parser.add_argument("--parallel-open-request-expansion-factor", default=1.5, type=numeric_min_checker(0.1, numeric_type=float, message="must have at least 0.1 expansion factor"),
                        help="the expansion factor to multiple the number of client slots by to bound the capacity in the global pipeline. Not used if parallel_open_requests is set")
    parser.add_argument("--credit-link", default=credit_link_successive, choices=(credit_link_end_to_end, credit_link_successive), help="Type of credit linking to use between successive stages")
    parser.add_argument("--align-counters", default=False, action="store_true", help="track the exit rate of the align stages")
    parser.add_argument("--sort-counters", default=False, action="store_true", help="track the exit rate of the sort stages")
    parser.add_argument("--merge-counters", default=False, action="store_true", help="track the exit rate of the merge stages")

class CephAlignSortUnfused(app.Application):

    ingress_dtypes = (tf.string,)
    ingress_shapes = ((2),)

    @staticmethod
    def name():
        return "ceph-align-sort-unfused"

    @staticmethod
    def help_message():
        return "align and sort a ceph dataset using Snap"

    class_logger = logging.getLogger(name="UnfusedCephAlignClass")

    @classmethod
    def _make_graph_args(cls, parser):
        add_common_args(parser=parser)
        parser.add_argument("--log-goodput", default=False, action='store_true', help="turn on all goodput and latency tracing")
        snap_align.CephSnapStage.add_graph_args(parser=parser)
        sort_stage.CephSort.add_graph_args(parser=parser)
        merge_stage.CephMergeStage.add_graph_args(parser=parser)

    @classmethod
    def device_counts(cls, args):
        return {
            align_key: args.align_stages,
            sort_key: args.sort_stages,
            merge_key: args.merge_stages,
        }

    def _construct_graph(self, args, device_map, num_client_slots):
        gate_name = "ingress_gate"

        num_align = args.align_stages
        num_sort = args.sort_stages
        num_merge = args.merge_stages
        align_devices = device_map[align_key]
        sort_devices = device_map[sort_key]
        merge_devices = device_map[merge_key]

        if args.parallel_open_requests is not None:
            capacity_between_gates = args.parallel_open_requests
        else:
            capacity_between_gates = int(num_client_slots * args.parallel_open_request_expansion_factor)
        if capacity_between_gates < 1:
            raise Exception("Capacity between gates is <1 ({c})".format(c=capacity_between_gates))
        args.parallel_open_requests = capacity_between_gates
        self.log.info("Capacity between gates: {}".format(capacity_between_gates))
        if args.credit_link == credit_link_end_to_end:
            inter_gate_kwargs = {
                "limit_upstream": False,
                "limit_downstream": False
            }
        else:
            inter_gate_kwargs = {
                "capacity": capacity_between_gates
            }

        def generate_local_pipelines(name, num_pipelines, devices, upstream_gate, pipeline_type):
            with tf.name_scope("{}_pipeline".format(name)):
                pipelines = tuple(pipeline_type(args=args) for _ in range(num_pipelines))
                assert len(pipelines) == len(devices)
                for pipeline, device in zip(pipelines, devices):
                    with device():
                        device_graph = pipeline.make_graph(upstream_gate=upstream_gate)
                    try: # convert to a tuple if it returns a generator
                        device_graph[0]
                    except TypeError:
                        device_graph = tuple(device_graph)
                    run_first = pipeline.run_first
                    for item in run_first:
                        self._add_run_first(item)
                    yield device_graph

        with tf.name_scope(gate_name):
            ingress = gate.IngressGate(dtypes=self.ingress_dtypes, shapes=self.ingress_shapes, capacity=capacity_between_gates,
                                       shared_name=gate_name, name=gate_name)

        align_outputs = tuple(generate_local_pipelines(
            name="align",
            devices=align_devices,
            upstream_gate=ingress,
            pipeline_type=snap_align.CephSnapStage,
            num_pipelines=num_align
        ))
        assert len(align_outputs) == num_align

        with tf.name_scope("sort_gate"):
            gate_name = "global_sort_gate"
            example_output = align_outputs[0]
            sort_gate = gate.StreamingGate(
                sample_tensors=example_output[4:6],
                id_and_count_upstream=example_output[0], join=True,
                name=gate_name, shared_name=gate_name,
                **inter_gate_kwargs
            )
            # 4 and 5 are key and namespace
            enqueue_ops = tuple(sort_gate.enqueue(id_and_count=a[0], components=a[4:6]) for a in align_outputs)

            if args.align_counters:
                if getattr(args, "summary", False):
                    with tf.name_scope(None):
                        with tf.name_scope(performance_name_scope):
                            enqueue_ops = tuple(make_counter(counter_name="aligned_counter",
                                                             summary_name="aligned_num_records",
                                                             deps_and_counters=zip(
                                                                 enqueue_ops,
                                                                 (a[3] for a in align_outputs)
                                                             )))
                else:
                    self.log.warning("Align counters requested, but no summary was requested. Please enable summary for this to work")
            gate.add_gate_runner(gate_runner=gate.GateRunner(gate=sort_gate, enqueue_ops=enqueue_ops, device=sort_gate.device))
            if args.credit_link == credit_link_successive:
                gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=sort_gate)

        sort_outputs = tuple(generate_local_pipelines(
            name="sort",
            devices=sort_devices,
            upstream_gate=sort_gate,
            pipeline_type=sort_stage.CephSort,
            num_pipelines=num_sort
        ))
        assert len(sort_outputs) == num_sort
        sort_outputs = tuple(itertools.chain.from_iterable(sort_outputs))

        with tf.name_scope("merge_gate"):
            gate_name = "global_merge_gate"
            example_output = sort_outputs[0]
            merge_gate = gate.StreamingGate(
                sample_tensors=example_output[1:-1], # slice off the id
                id_and_count_upstream=example_output[0], join=True,
                name=gate_name, shared_name=gate_name,
                **inter_gate_kwargs
            )
            enqueue_ops = tuple(merge_gate.enqueue(id_and_count=a[0], components=a[1:-1]) for a in sort_outputs)

            if args.align_counters:
                if getattr(args, "summary", False):
                    with tf.name_scope(None):
                        with tf.name_scope(performance_name_scope):
                            enqueue_ops = tuple(make_counter(counter_name="sorted_counter",
                                                             summary_name="sorted_num_records",
                                                             deps_and_counters=zip(
                                                                 enqueue_ops,
                                                                 (a[3] for a in sort_outputs)
                                                             )))
                else:
                    self.log.warning("Sort counters requested, but no summary was requested. Please enable summary for this to work")
            gate.add_gate_runner(gate_runner=gate.GateRunner(gate=merge_gate, enqueue_ops=enqueue_ops, device=merge_gate.device))
            if args.credit_link == credit_link_successive:
                gate.add_credit_supplier_from_gates(upstream_gate=sort_gate, downstream_gate=merge_gate)

        merge_outputs = tuple(generate_local_pipelines(
            name="merge",
            devices=merge_devices,
            upstream_gate=merge_gate,
            pipeline_type=merge_stage.CephMergeStage,
            num_pipelines=num_merge
        ))
        assert len(merge_outputs) == num_merge
        merge_outputs = tuple(itertools.chain.from_iterable(merge_outputs))

        example_output = merge_outputs[0]
        gate_name = "egress_gate"
        with tf.name_scope(gate_name):
            egress = gate.EgressGate(capacity=capacity_between_gates, sample_tensors=example_output.components,
                                     id_and_count_upstream=example_output.id_and_count, join=True,
                                     name=gate_name, shared_name=gate_name)
            enqueue_ops = tuple(egress.enqueue(id_and_count=a.id_and_count, components=a.components) for a in merge_outputs)
            if args.merge_counters:
                if getattr(args, "summary", False):
                    with tf.name_scope(None):
                        with tf.name_scope(performance_name_scope):
                            enqueue_ops = tuple(make_counter(counter_name="merged_counter",
                                                             summary_name="merged_num_records",
                                                             deps_and_counters=zip(
                                                                 enqueue_ops,
                                                                 (a.components[2] for a in merge_outputs)
                                                             )))
                else:
                    self.log.warning("Merge counters requested, but no summary was requested. Please enable summary for this to work")

            gate.add_gate_runner(gate_runner=gate.GateRunner(gate=egress, enqueue_ops=enqueue_ops,
                                                             device=egress.device))
            if args.credit_link == credit_link_end_to_end:
                gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=egress)
            else:
                gate.add_credit_supplier_from_gates(upstream_gate=merge_gate, downstream_gate=egress)

        self.close_op = (ingress.close(), sort_gate.close(), merge_gate.close(), egress.close())
        self.close_op = self.close_op[:1]
        self.close_op = []

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
        parser.add_argument("--overwrite", default=False, action="store_true", help="Overwrite existing metadata file when the pipeline finishes. Default: create a new one")
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
        def get_result_columns(key_columns):
            a = set()
            for key_column in key_columns:
                decoded_key = key_column[0]
                split = decoded_key.split(".")
                assert len(split) == 2
                extension = split[1]
                if extension in a:
                    raise Exception("Runtime error: extension '{e}' already in a column. Columns: {c}".format(
                        e=extension, c=", ".join(x for x in a)
                    ))
                a.add(extension)
                for other_column in key_column[1:]:
                    other_decoded_key = other_column
                    other_split = other_decoded_key.split(".")
                    assert len(other_split) == 2
                    other_extension = other_split[1]
                    if other_extension != extension:
                        raise Exception("Expected all column extensions to match. Expected '{exp}', but got '{actual}'".format(
                            exp=extension, actual=other_extension
                        ))
            return a

        record_ids, first_ordinals, num_recordz, keys, namespaces = results[:5]
        full_keys_records = results[5:]
        dataset = args.dataset
        output_filepath = dataset.pop(filepath_key)
        processed_columns = get_result_columns(key_columns=full_keys_records)
        columns = set(dataset["columns"])
        if not columns.issubset(processed_columns):
            raise Exception("Expected more columns, but got fewer. Before: [{before}], After: [{after}]".format(
                before=", ".join(columns), after=", ".join(processed_columns)
            ))
        if snap_common.results_extension not in processed_columns:
            raise Exception("Expected extension '{res_ext}' in the results extensions, but didn't find it.".format(res_ext=snap_common.results_extension))
        only_secondary = processed_columns.difference({snap_common.results_extension,
                                                       snap_common.base_extension,
                                                       snap_common.metadata_extension,
                                                       snap_common.qual_extension})
        for res_ext in only_secondary:
            stripped = res_ext.translate({ord(k): None for k in digits})
            if stripped != snap_common.secondary_results_extension:
                raise Exception("Secondary or unknown results extension found: '{found}'".format(found=res_ext))

        merge_att = "_".join((merge_stage.CephMergeStage.local_dest, "make_new"))
        overwrite = args.overwrite or (hasattr(args, merge_att) and not getattr(args, merge_att))
        if not overwrite:
            filename = output_filepath.stem
            file_dir = output_filepath.parent
            output_filepath = file_dir / (filename+"_sorted"+output_filepath.suffix)
        if output_filepath.exists():
            cls.class_logger.warning("Output metadata path '{p}' exists. Will overwrite!".format(p=str(output_filepath)))

        columns_to_add = sorted(tuple(processed_columns.difference(columns)))
        dataset["columns"].extend(columns_to_add)

        first_record_id = record_ids[0]
        if not all(f == first_record_id for f in record_ids[1:]):
            raise Exception("Not all record IDs are equal: [{rids}]".format(rids=", ".join(record_ids)))

        if first_record_id != dataset["name"]:
            cls.class_logger.warning("Input metadata specified a record id of '{metadata_version}', but pipeline output '{new_version}'. Overwriting with {new_version}.".format(
                metadata_version=dataset["name"], new_version=first_record_id
            ))
            dataset["name"] = first_record_id

        new_records = [
            {
                "path": path,
                "first": first_ordinal,
                "last": first_ordinal + num_records
            } for path, first_ordinal, num_records in zip(keys, first_ordinals, num_recordz)
        ]
        dataset["records"] = new_records

        with output_filepath.open("w+") as f:
            json.dump(dataset, f, indent=4)

    def _run_client_request(self, client_args, client_slot, sess):
        client_args = tuple(client_args)
        ingress_placeholder = client_slot.ingress_placeholders[0]
        egress_dequeue = client_slot.egress_dequeue
        results = sess.run(egress_dequeue, feed_dict={ingress_placeholder: client_args})
        record_ids, first_ordinals, num_recordz, keys, namespaces = results[:5]
        full_keys_records = results[5:]
        utf8 = "utf-8"
        new_record_ids = tuple(i.decode(utf8) for i in record_ids)
        new_first_ordinals = tuple(int(i) for i in first_ordinals)
        new_num_recordz = tuple(int(i) for i in num_recordz)
        new_keys = tuple(i.decode(utf8) for i in keys)
        new_namespaces = tuple(i.decode(utf8) for i in namespaces)
        new_full_keys_records = tuple(
            tuple(b.decode(utf8) for b in ffp)
            for ffp in full_keys_records
        )
        return (new_record_ids, new_first_ordinals, new_num_recordz, new_keys, new_namespaces) + new_full_keys_records

    def stop(self, sess):
        for o in self.close_op:
            try:
                sess.run(o)
            except Exception as e:
                self.log.error("{nm} closing. Got exception '{e}' when running op '{on}'".format(on=o.name, e=e, nm=self.name()))
