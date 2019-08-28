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
from modules.snap_align import snap_align as stage
from common.parse import numeric_min_checker, add_dataset, path_exists_checker, filepath_key
import tensorflow.contrib.gate as gate
import tensorflow as tf
import logging; logging.basicConfig(level=logging.DEBUG)
import json
from .common import make_counter

device_type_name = "align"

def add_common_arguments(parser):
    parser.add_argument("--align-counters", default=False, action="store_true", help="track the exit rate of the align/sort stages")
    parser.add_argument("--align-stages", dest="stages", default=1, type=numeric_min_checker(1, "must have at least 1 fused_align_sort"), help="number of align stages")
    parser.add_argument("--parallel-open-requests", type=numeric_min_checker(1, "must have at least 1 parallel open request"), help="if specified, the number of parallel open requests")
    parser.add_argument("--parallel-open-request-expansion-factor", default=1.5, type=numeric_min_checker(0.1, numeric_type=float, message="must have at least 0.1 expansion factor"),
                        help="the expansion factor to multiple the number of client slots by to bound the capacity in the global pipeline. Not used if parallel_open_requests is set")

class Align(app.Application):

    ingress_dtypes = (tf.string,)
    ingress_shapes = ((),)

    @staticmethod
    def name():
        return "align"

    @staticmethod
    def help_message():
        return "align a dataset using Snap"

    class_logger = logging.getLogger(name="AlignClass")
    class_logger.setLevel(level=logging.DEBUG)

    @classmethod
    def _make_graph_args(cls, parser):
        # TODO need to do the subparsers thing here when there ceph option is available
        add_common_arguments(parser=parser)
        parser.add_argument("--log-goodput", default=False, action='store_true', help="turn on all goodput and latency tracing")
        stage.LocalSnapStage.add_graph_args(parser=parser)

    @classmethod
    def device_counts(cls, args):
        return { device_type_name: args.stages }

    def _construct_graph(self, args, device_map, num_client_slots):
        # need to set ingress and egress queue
        devices = device_map[device_type_name]
        num_devices = len(devices)

        stages = tuple(stage.LocalSnapStage(args=args) for _ in range(num_devices))

        gate_name = "ingress_gate"
        if args.parallel_open_requests is not None:
            capacity_between_gates = args.parallel_open_requests
        else:
            capacity_between_gates = int(num_client_slots * args.parallel_open_request_expansion_factor)
        if capacity_between_gates < 1:
            raise Exception("Capacity between gates is <1 ({c})".format(c=capacity_between_gates))
        ingress = gate.IngressGate(dtypes=self.ingress_dtypes, shapes=self.ingress_shapes, capacity=capacity_between_gates,
                                   shared_name=gate_name, name=gate_name)

        def make_stages():
            for stage, device in zip(stages, devices):
                with device():
                    device_graph = stage.make_graph(upstream_gate=ingress)
                    try: # convert to a tuple if it returns a generator
                        device_graph[0]
                    except TypeError:
                        device_graph = tuple(device_graph)
                    run_first = stage.run_first
                    assert len(run_first) > 0
                    for item in run_first:
                        self._add_run_first(item)
                    yield device_graph
        with tf.name_scope("align_pipeline"):
            outputs = tuple(make_stages())
        assert len(outputs) == len(stages)

        example_output = outputs[0]
        egress = gate.EgressGate(capacity=capacity_between_gates, sample_tensors=example_output[1:], id_and_count_upstream=example_output[0], join=True)
        enqueue_ops = tuple(egress.enqueue(id_and_count=a[0], components=a[1:]) for a in outputs)

        if args.align_counters:
            if getattr(args, "summary", False):
                with tf.name_scope(None):
                    with tf.name_scope("performance"):
                        enqueue_ops = tuple(make_counter(counter_name="aligned_counter",
                                                         summary_name="aligned_num_records",
                                                         deps_and_counters=zip(
                                                             enqueue_ops,
                                                             (a[3] for a in outputs)
                                                         )))
            else:
                self.log.warning("Align counters requested, but no summary was requested. Please enable summary for this to work.")

        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=egress, enqueue_ops=enqueue_ops, device=egress.device))
        gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=egress)

        self.close_op = egress.close()

        unknown_shape = tf.TensorShape([None])
        batch_ingress_shapes = tuple(unknown_shape.concatenate(ishape) for ishape in self.ingress_shapes)
        for _ in range(num_client_slots):
            ingress_placeholders = tuple(tf.placeholder(dtype=dtype, shape=shape) for dtype, shape in zip(self.ingress_dtypes, batch_ingress_shapes))
            ingress_enqueue = ingress.enqueue_request(components=ingress_placeholders)
            egress_dequeue = egress.dequeue_request(request_id=ingress_enqueue)
            yield self.ClientSlot(ingress_placeholders=ingress_placeholders, egress_dequeue=egress_dequeue)

    @classmethod
    def make_client_args(cls, parser):
        # TODO assume that for now it is just the local filesystem. Will need to differentiate for other stuff later
        add_dataset(parser=parser)
        parser.add_argument("-d", "--dataset-dir", type=path_exists_checker(), help="Directory containing ALL of the chunk files")

    @classmethod
    def process_ingress_args(cls, args):
        dataset_dir = args.dataset_dir
        if dataset_dir is None:
            metadata_path = args.dataset[filepath_key]
            dataset_dir = metadata_path.parent
        files_to_remove = tuple(itertools.chain(dataset_dir.glob("*.results"), dataset_dir.glob("*.secondary*")))
        if len(files_to_remove) > 0:
            cls.class_logger.info("Removing prior results before aligning: {}".format(files_to_remove))
            for f in files_to_remove:
                assert f.is_file()
                f.unlink()
        if len(args.dataset["records"]) == 0:
            raise ValueError("Dataset must have non-zero number of records")
        return (dataset_dir / record["path"] for record in args.dataset["records"])

    @staticmethod
    def parse_and_verify_results(results):
        record_ids = results[0]
        record_id_count = len(record_ids)
        assert record_id_count > 0
        first_record_id = record_ids[0]
        assert all(rid == first_record_id for rid in record_ids)
        first_ordinals = results[1]
        assert len(first_ordinals) == record_id_count
        num_records = results[2]
        assert len(num_records) == record_id_count
        file_basenames = results[3]
        assert len(file_basenames) == record_id_count
        result_filenames = results[4:]
        assert len(result_filenames) > 0
        assert all(len(r) == record_id_count for r in result_filenames)

        result_filename_column = result_filenames[0]
        extensions = set()
        for basename, result_column_name in zip(file_basenames, result_filename_column):
            column_basename, extension = result_column_name.rsplit(".", 1)
            assert extension == "results"
            assert column_basename == basename
            if extension not in extensions:
                extensions.add(extension)

        for index, secondary_column in enumerate(result_filenames[1:]):
            expected_column_extension = "secondary{}".format(index)
            extensions.add(expected_column_extension)
            for basename, result_column_name in zip(file_basenames, secondary_column):
                column_basename, extension = result_column_name.rsplit(".", 1)
                assert extension == expected_column_extension
                assert column_basename == basename

        return first_record_id, first_ordinals, num_records, file_basenames, extensions

    @classmethod
    def process_egress_results(cls, results, args):
        """
        :param results: a list of [ record_id, first_ordinal, num_records, file_basename, written_records], where written_records is a list of results, then all the secondary files (all strings)
        :param args:
        :return:
        """
        record_id, first_ordinals, num_records, file_basenames, extensions = cls.parse_and_verify_results(results=results)

        output_filepath = args.dataset.pop(filepath_key)
        columns = args.dataset["columns"]
        for extension in sorted(extensions): # will put results first, then all secondary
            if extension not in columns:
                columns.append(extension)
        with output_filepath.open("w+") as f:
            json.dump(args.dataset, f, indent=4)

    def _run_client_request(self, client_args, client_slot, sess):
        client_args = tuple(client_args)
        ingress_placeholder = client_slot.ingress_placeholders[0]
        egress_dequeue = client_slot.egress_dequeue
        results = sess.run(egress_dequeue, feed_dict={ingress_placeholder: tuple(str(c) for c in client_args)})
        record_ids, first_ordinals, num_recordz, file_basenames = results[:4]
        full_file_pathz = tuple(results[4:])
        utf8 = "utf-8"
        new_record_ids = tuple(i.decode(utf8) for i in record_ids)
        new_first_ordinals = tuple(int(i) for i in first_ordinals)
        new_num_recordz = tuple(int(i) for i in num_recordz)
        new_file_basenames = tuple(i.decode(utf8) for i in file_basenames)
        new_full_file_pathz = tuple(
            tuple(b.decode(utf8) for b in ffp)
            for ffp in full_file_pathz
        )
        return (new_record_ids, new_first_ordinals, new_num_recordz, new_file_basenames) + new_full_file_pathz

    def stop(self, sess):
        try:
            sess.run(self.close_op)
        except Exception as e:
            self.log.error("{nm} closing. Got exception '{e}'".format(e=e, nm=self.name()))

class CephAlign(app.Application):

    ingress_dtypes = (tf.string,)
    ingress_shapes = ((2),)

    @staticmethod
    def name():
        return "ceph-align"

    @staticmethod
    def help_message():
        return "align a dataset using Snap on a ceph filesystem"

    class_logger = logging.getLogger(name="CephAlignClass")
    class_logger.setLevel(level=logging.DEBUG)

    @classmethod
    def _make_graph_args(cls, parser):
        add_common_arguments(parser=parser)
        parser.add_argument("--log-goodput", default=False, action='store_true', help="turn on all goodput and latency tracing")
        stage.CephSnapStage.add_graph_args(parser=parser)

    @classmethod
    def device_counts(cls, args):
        return { device_type_name: args.stages }

    def _construct_graph(self, args, device_map, num_client_slots):
        devices = device_map[device_type_name]
        num_devices = len(devices)

        gate_name = "ingress_gate"
        if args.parallel_open_requests is not None:
            capacity_between_gates = args.parallel_open_requests
        else:
            capacity_between_gates = int(num_client_slots * args.parallel_open_request_expansion_factor)
        if capacity_between_gates < 1:
            raise Exception("Capacity between gates is <1 ({c})".format(c=capacity_between_gates))
        ingress = gate.IngressGate(dtypes=self.ingress_dtypes, shapes=self.ingress_shapes, capacity=capacity_between_gates,
                                   shared_name=gate_name, name=gate_name)

        with tf.name_scope("align_pipeline"):
            stages = tuple(stage.CephSnapStage(args=args) for _ in range(num_devices))
            def make_stages():
                for stage, device in zip(stages, devices):
                    with device():
                        device_graph = stage.make_graph(upstream_gate=ingress)
                        try: # convert to a tuple if it returns a generator
                            device_graph[0]
                        except TypeError:
                            device_graph = tuple(device_graph)
                        run_first = stage.run_first
                        assert len(run_first) > 0
                        for item in run_first:
                            self._add_run_first(item)
                        yield device_graph
            outputs = tuple(make_stages())
        assert len(outputs) == len(stages)

        example_output = outputs[0]
        egress = gate.EgressGate(capacity=capacity_between_gates, sample_tensors=example_output[1:], id_and_count_upstream=example_output[0], join=True)
        enqueue_ops = tuple(egress.enqueue(id_and_count=a[0], components=a[1:]) for a in outputs)

        if args.align_counters:
            if getattr(args, "summary", False):
                with tf.name_scope(None):
                    with tf.name_scope("performance"):
                        enqueue_ops = tuple(make_counter(counter_name="aligned_counter",
                                                         summary_name="aligned_num_records",
                                                         deps_and_counters=zip(
                                                             enqueue_ops,
                                                             (a[3] for a in outputs)
                                                         )))
            else:
                self.log.warning("Align counters requested, but no summary was requested. Please enable summary for this to work.")

        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=egress, enqueue_ops=enqueue_ops, device=egress.device))
        gate.add_credit_supplier_from_gates(upstream_gate=ingress, downstream_gate=egress)

        self.close_op = egress.close()

        unknown_shape = tf.TensorShape([None])
        batch_ingress_shapes = tuple(unknown_shape.concatenate(ishape) for ishape in self.ingress_shapes)
        for _ in range(num_client_slots):
            ingress_placeholders = tuple(tf.placeholder(dtype=dtype, shape=shape) for dtype, shape in zip(self.ingress_dtypes, batch_ingress_shapes))
            ingress_enqueue = ingress.enqueue_request(components=ingress_placeholders)
            egress_dequeue = egress.dequeue_request(request_id=ingress_enqueue)
            yield self.ClientSlot(ingress_placeholders=ingress_placeholders, egress_dequeue=egress_dequeue)

    @classmethod
    def make_client_args(cls, parser):
        parser.add_argument("--namespace", default="", help="the namespace to access this dataset")
        parser.add_argument("--use-default-namespace", default=False, action="store_true", help="use the name of this record as the namespace")
        add_dataset(parser=parser)

    @classmethod
    def process_ingress_args(cls, args):
        dataset = args.dataset
        if args.use_default_namespace:
            namespace = dataset["name"]
        else:
            namespace = args.namespace
        record_keys = (a["path"] for a in dataset["records"])
        return tuple(zip(record_keys, itertools.repeat(namespace)))

    @staticmethod
    def parse_and_verify_results(results):
        record_ids = results[0]
        record_id_count = len(record_ids)
        assert record_id_count > 0
        first_record_id = record_ids[0]
        assert all(rid == first_record_id for rid in record_ids)
        first_ordinals = results[1]
        assert len(first_ordinals) == record_id_count
        num_records = results[2]
        assert len(num_records) == record_id_count
        file_keys = results[3]
        assert len(file_keys) == record_id_count
        namespaces = results[4]
        assert len(namespaces) == record_id_count

        first_namespace = namespaces[0]
        assert all(n == first_namespace for n in namespaces[1:])

        result_filenames = results[5:]
        assert len(result_filenames) > 0
        assert all(len(r) == record_id_count for r in result_filenames)

        result_filename_column = result_filenames[0]
        extensions = set()
        for basename, result_column_name in zip(file_keys, result_filename_column):
            column_basename, extension = result_column_name.rsplit(".", 1)
            assert extension == "results"
            assert column_basename == basename
            if extension not in extensions:
                extensions.add(extension)

        for index, secondary_column in enumerate(result_filenames[1:]):
            expected_column_extension = "secondary{}".format(index)
            extensions.add(expected_column_extension)
            for basename, result_column_name in zip(file_keys, secondary_column):
                column_basename, extension = result_column_name.rsplit(".", 1)
                assert extension == expected_column_extension
                assert column_basename == basename

        return first_record_id, first_ordinals, num_records, file_keys, first_namespace, extensions

    @classmethod
    def process_egress_results(cls, results, args):
        """
        :param results: a list of [ record_id, first_ordinal, num_records, file_basename, written_records], where written_records is a list of results, then all the secondary files (all strings)
        :param args:
        :return:
        """
        record_id, first_ordinals, num_records, file_keys, namespace, extensions = cls.parse_and_verify_results(results=results)

        output_filepath = args.dataset.pop(filepath_key)
        columns = args.dataset["columns"]
        for extension in sorted(extensions): # will put results first, then all secondary
            if extension not in columns:
                columns.append(extension)
        with output_filepath.open("w+") as f:
            json.dump(args.dataset, f, indent=4)

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
        try:
            sess.run(self.close_op)
        except Exception as e:
            self.log.error("{nm} closing. Got exception '{e}'".format(e=e, nm=self.name()))
