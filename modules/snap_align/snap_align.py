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
import multiprocessing
from ..common.stage import Stage
from common.parse import numeric_min_checker, path_exists_checker, non_empty_string_checker, yes_or_no
from .common import Ceph, sanitize_generator, slice_id, base_extension, qual_extension

import tensorflow as tf
from abc import abstractmethod

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import pipeline
import tensorflow.contrib.gate as gate

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)

class SnapCommonStage(Stage):
    columns = [base_extension, qual_extension]
    local_dest = "align"

    def __init__(self, args):
        super().__init__()
        self._run_first = []
        expected_args = ("read_parallel", "decompress_parallel", "align_parallel", "aligner_threads", "compress_parallel",
                         "write_parallel", "deep_verify", "paired", "snap_args", "subchunking", "index_path", "max_secondary",
                         "global_batch", "log_goodput", "log_directory")
        for expected_arg in expected_args:
            arg_name = "_".join((self.local_dest, expected_arg))
            setattr(self, expected_arg, getattr(args, arg_name))

        if hasattr(args, "log_goodput") and args.log_goodput is True:
            self.log.info("Override enabling log goodput from global param")
            self.log_goodput = args.log_goodput

        write_columns = ["results"]
        write_columns.extend("secondary{}".format(i) for i in range(self.max_secondary))
        self.write_columns = [ {"type": "structured", "extension": a} for a in write_columns]

        queue_length_defaults = (
            # These default values should be small because they're ahead of the core (the aligner)
            ("pre_decomp_capacity", "decompress_parallel", 1),
            ("pre_align_capacity", "align_parallel", 1.5),

            # these don't matter as much because they're after the core and will basically be empty
            ("pre_compress_capacity", "compress_parallel", 1),
            ("pre_write_capacity", "write_parallel", 1),
            ("final_sink_capacity", "write_parallel", 1)
        )

        for queue_cap_attr, source_attr, expansion_factor in queue_length_defaults:
            args_attr = "_".join((self.local_dest, queue_cap_attr))
            args_value = getattr(args, args_attr)
            default_value = int(getattr(self, source_attr) * expansion_factor)
            if args_value is None:
                args_value = default_value
                setattr(args, args_attr, args_value) # set this again so runtime can write this out correctly
            elif args_value < default_value:
                log.warning("Setting the queue capacity '{name}' to {set}. Recommended minimum is {rec}".format(
                    name=queue_cap_attr, set=args_value, rec=default_value
                ))
            setattr(self, queue_cap_attr, args_value)

    @property
    def run_first(self):
        return self._run_first

    @classmethod
    def add_common_graph_args(cls, parser):
        prefix=cls.local_dest
        cls.prefix_option(parser=parser, prefix=prefix, argument="read-parallel", type=numeric_min_checker(1, "must have >0 parallel read stages"), default=2, help="number of read stages to run in parallel")
        cls.prefix_option(parser=parser, prefix=prefix, argument="decompress-parallel", type=numeric_min_checker(1, "must have >0 parallel decomp stages"), default=3, help="number of decompress stages to run in parallel")
        cls.prefix_option(parser=parser, prefix=prefix, argument="align-parallel", type=numeric_min_checker(1, "must have >0 parallel align stages"), default=8, help="number of parallel align stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="aligner-threads", type=numeric_min_checker(1, "must have >0 parallel aligner threads"), default=multiprocessing.cpu_count()-2, help="number of aligner threads for shared aligner")
        cls.prefix_option(parser=parser, prefix=prefix, argument="compress-parallel", type=numeric_min_checker(1, "must have >0 parallel compress stages"), default=2, help="number of parallel compress stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="write-parallel", type=numeric_min_checker(1, "must have >0 parallel write stages"), default=2, help="number of parallel write stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="deep-verify", default=False, action='store_true', help="verify record integrity")
        cls.prefix_option(parser=parser, prefix=prefix, argument="paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
        cls.prefix_option(parser=parser, prefix=prefix, argument="snap-args", type=str, default="", help="SNAP algorithm specific self. Pass with enclosing \" \". E.g. \"-om 5 -omax 1\" . See SNAP documentation for all options.")
        cls.prefix_option(parser=parser, prefix=prefix, argument="subchunking", type=numeric_min_checker(100, "don't go lower than 100 for subchunking size"), default=5000, help="the size of each subchunk (in number of reads)")
        # Note: can't have path-exists checker for this because the path might be on a remote machine
        cls.prefix_option(parser=parser, prefix=prefix, argument="index-path", default="/home/whitlock/tf/ref_index", help="location of the ref index on all machines. Make sure all machines have this path!")
        cls.prefix_option(parser=parser, prefix=prefix, argument="max-secondary", type=numeric_min_checker(0, "must have a non-negative number of secondary results"), default=0, help="Max secondary results to store. >= 0 ")
        cls.prefix_option(parser=parser, prefix=prefix, argument="global-batch", type=numeric_min_checker(1, "must have >=1 batch from global gate"), default=2, help="batch size for dequeuing from the upstream central gate. Doesn't affect correctness")

        # all options below here are rather verbose, for length of queues
        # cls.prefix_option(parser=parser, prefix=prefix, argument="head-gate-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of capacity for head gate")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-decomp-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-read, pre-decomp queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-align-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-decomp, pre-align queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-compress-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-align, pre-compress queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-write-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-align, pre-write queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="final-sink-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity of final queue of this stage")

        cls.prefix_option(parser=parser, prefix=prefix, argument="log-goodput", default=False, action="store_true", help="log the goodput events")
        cls.prefix_option(parser=parser, prefix=prefix, argument="log-directory", default="/home/whitlock/tf/shell", help="the directory to log all events to, if log_goodput is enabled")

    def make_central_pipeline(self, inputs):
        """
        Make the central pipeline between the custom read and write operations
        :param args:
        :param inputs: a generator of type (id_and_count, column0, column1, ..., [:rest of input]). The number of colums is assumed to be the same and in the same order as self.columns
        :return: a generator of [ compressed_results_column_matrix, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        """

        if not isinstance(inputs, (list, tuple)):
            inputs = tuple(inputs)

        # type of each of these: (id_and_count, column0, column1, ..., [:rest of input])
        queue_name = "align_ready_to_decomp"
        ready_to_decomp = pipeline.join(upstream_tensors=inputs,
                                        parallel=self.decompress_parallel,
                                        capacity=self.pre_decomp_capacity, multi=True,
                                        name=queue_name, shared_name=queue_name)
        with tf.name_scope("decompression_stage"):
            ready_to_align_items = self.make_decomp_stage(ready_to_decomp=ready_to_decomp)

        queue_name = "ready_to_align"
        ready_to_align = pipeline.join(upstream_tensors=ready_to_align_items,
                                       parallel=self.align_parallel,
                                       capacity=self.pre_align_capacity, multi=True,
                                       name=queue_name, shared_name=queue_name)

        with tf.name_scope("align_stage"):
            ready_to_compress_items = self.make_align_stage(ready_to_align=ready_to_align)

        queue_name = "align_ready_to_compress"
        ready_to_compress = pipeline.join(upstream_tensors=ready_to_compress_items,
                                          parallel=self.compress_parallel,
                                          capacity=self.pre_compress_capacity, multi=True,
                                          name=queue_name, shared_name=queue_name)

        with tf.name_scope("compress_stage"):
            ready_to_write_items = tuple(self.make_compress_stage(ready_to_compress=ready_to_compress))

        def gen_control_deps():
            for item in ready_to_write_items:
                num_records, ordinal, record_id = item[1:4]
                item_id = slice_id(item[4])
                with tf.control_dependencies((item_id,)):
                    ts = gate.unix_timestamp(name="align_tail_timestamp")
                yield (gate.log_events(
                    item_names=("id", "time", "ordinal", "record_id", "num_records"),
                    directory=self.log_directory,
                    event_name="align_tail",
                    name="align_tail_event_logger",
                    components=(item_id, ts, ordinal, record_id, num_records)
                ),)

        control_deps = []
        if self.log_goodput:
            control_deps.extend(gen_control_deps())

        queue_name = "ready_to_write"
        ready_to_write = pipeline.join(upstream_tensors=ready_to_write_items,
                                       control_dependencies=control_deps,
                                       parallel=self.write_parallel,
                                       capacity=self.pre_write_capacity, multi=True,
                                       name=queue_name, shared_name=queue_name)
        return ready_to_write

    @staticmethod
    def make_compress_stage(ready_to_compress):
        """
        :param ready_to_compress: a generator of [ results_column_matrix, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        :return: a generator of [ compressed_results_column_matrix, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        """
        to_compress_gen, around_compress_gen = zip(*(
            (a[0], a[1:]) for a in ready_to_compress
        ))
        compressed_buffers = pipeline.aligner_compress_pipeline(upstream_tensors=to_compress_gen)
        return ((a,) + tuple(b) for a,b in zip(compressed_buffers, around_compress_gen))

    def make_align_stage(self, ready_to_align):
        """
        :param args:
        :param ready_to_align: a generator of [ agd_read_handle, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        :return: a generator of [ results_column_matrix, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        """
        to_aligners, around_aligners = zip(*(
            (a[0], a[1:]) for a in ready_to_align
        ))

        if self.paired:
            aligner_type = persona_ops.snap_align_paired
            aligner_options = persona_ops.paired_aligner_options(cmd_line=self.snap_args.split(), name="paired_aligner_options")
            executor_type = persona_ops.snap_paired_executor
        else:
            aligner_type = persona_ops.snap_align_single
            aligner_options = persona_ops.aligner_options(cmd_line=self.snap_args.split(), name="aligner_options") # -o output.sam will not actually do anything
            executor_type = persona_ops.snap_single_executor

        buffer_list_pool = persona_ops.buffer_list_pool(**pipeline.pool_default_args)
        genome = persona_ops.genome_index(genome_location=self.index_path, name="genome_loader", shared_name="genome_loader")
        self._run_first.append(genome)

        single_executor = executor_type(num_threads=self.aligner_threads,
                                        work_queue_size=int(self.aligner_threads),
                                        options_handle=aligner_options,
                                        genome_handle=genome)

        aligner_results = (aligner_type(
            read=read_handle,
            buffer_list_pool=buffer_list_pool,
            subchunk_size=self.subchunking,
            executor_handle=single_executor,
            max_secondary=self.max_secondary
        ) for read_handle in to_aligners)

        for aligner_result, around_aligner in zip(aligner_results, around_aligners):
            yield (aligner_result,) + tuple(around_aligner)

    def make_decomp_stage(self, ready_to_decomp):
        """
        :param args:
        :param ready_to_decomp: generator of (id_and_count, column0, column1, ..., [:rest of input])
        :return: a generator of [ agd_read_handle, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        """
        ready_to_decomp = sanitize_generator(ready_to_decomp)
        num_columns = len(self.columns)

        # to_agd_reader = just the columns
        # pass_around_agd_reader = (id_and_count, rest, of, input, ...)
        to_agd_reader, pass_around_agd_reader = zip(*(
            (rtd[1:1+num_columns], (rtd[0],)+tuple(rtd[1+num_columns:])) for rtd in ready_to_decomp
        ))

        def gen_timestamps():
            for group in pass_around_agd_reader:
                with tf.control_dependencies((group[0],)):
                    yield gate.unix_timestamp(name="align_head_timestamp")

        reader_kwargs = {}
        timestamps = []
        if self.log_goodput:
            timestamps.extend(gen_timestamps())
            assert len(timestamps) == len(ready_to_decomp)
            # control dependencies have to be an iterable
            reader_kwargs["control_ops"] = tuple((a,) for a in timestamps)

        # [output_buffer_handles], num_records, first_ordinal, record_id; in order, for each column group in upstream_tensorz
        multi_column_gen = tuple(pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_agd_reader, verify=self.deep_verify,
                                                                           name="align_reader", **reader_kwargs))

        # around = num_records, first_ordinal, record_id for each group
        to_assembler, around_assembler = zip(*(
            (a[:2], a[1:]) for a in multi_column_gen
        ))

        assembler_kwargs = {}
        if self.log_goodput:
            log_event_ops = [
                (gate.log_events( # single element tuple because that's how tf.control_dependencies works
                    item_names=("id","time","ordinal","record_id"),
                    components=(in_id, timestamp, ordinal, record_id),
                    event_name="align_head",
                    directory=self.log_directory,
                    name="align_head_event_logger"
                ),) for in_id, timestamp, ordinal, record_id in zip(
                    (slice_id(a[0]) for a in pass_around_agd_reader),
                    timestamps,
                    (b[2] for b in multi_column_gen),
                    (b[3] for b in multi_column_gen)
                )
            ]
            assembler_kwargs["control_deps"] = log_event_ops

        # each element is an agd_reads handle
        agd_assembled_reads = pipeline.agd_read_assembler(upstream_tensors=to_assembler, include_meta=False, **assembler_kwargs)
        for agd_read, around_assembler_group, around_reader_group in zip(agd_assembled_reads, around_assembler, pass_around_agd_reader):
            yield (agd_read,) + tuple(around_assembler_group) + tuple(around_reader_group)

    @abstractmethod
    def make_graph_impl(self, local_gate):
        raise NotImplementedError

    def make_head_gate(self, upstream_gate):
        id_and_count, components = upstream_gate.dequeue_partition(count=self.global_batch)
        gate_name = "_".join((self.local_dest, "head_gate"))
        head_gate = gate.StreamingGate(limit_upstream=False, limit_downstream=False, # turning both off because there is only one needed, and no credit control is necessary
                                       id_and_count_upstream=id_and_count, sample_tensors=components,
                                       sample_tensors_are_batch=True,
                                       capacity=2,
                                       name=gate_name, shared_name=gate_name)
        enq_ops = (head_gate.enqueue_many(id_and_count=id_and_count, components=components),)
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=head_gate, enqueue_ops=enq_ops))
        return head_gate

    def _make_graph(self, upstream_gate):
        head_gate = self.make_head_gate(upstream_gate=upstream_gate)
        return self.make_graph_impl(local_gate=head_gate)

class LocalSnapStage(SnapCommonStage):

    @classmethod
    def add_graph_args(cls, parser):
        cls.add_common_graph_args(parser=parser)
        cls.prefix_option(parser=parser, prefix=cls.local_dest, argument="path-prefix", default="", help="path prefix to assign to this fused_align_sort, for example for a common FUSE mount point")

    def __init__(self, args):
        super().__init__(args)
        self.path_prefix = getattr(args, "{}_path_prefix".format(self.local_dest))

    def make_read_stage(self, gate):
        """
        :param gate:
        :param args:
        :return: a generator of [ id_and_count, [ filename ], [ a list of handles in the order of the columns, NOT STACKED ] ]
        """
        # each item in dequeue_ops' components is a single filename
        dequeue_ops = tuple(gate.dequeue() for _ in range(self.read_parallel))
        filenames = (components[0] for _, components in dequeue_ops)
        path_prefix = self.path_prefix
        if path_prefix != "":
            if path_prefix[-1] != "/":
                path_prefix = "{}/".format(path_prefix)
            path_prefix = tf.constant(path_prefix)
            filenames = (tf.string_join((path_prefix, fname)) for fname in filenames)
        read_file_gen = zip(dequeue_ops, pipeline.local_read_pipeline(
            upstream_tensors=filenames, columns=self.columns # a[1][0] gets the components, which is just a filename
        ))
        for a,b in read_file_gen:
            yield tuple(a)+tuple(b)

    def make_write_stage(self, write_ready_inputs):
        """
        :param args:
        :param write_ready_inputs: a generator of [ compressed_results_column_matrix, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        :return: a gen of [ id_and_count, record_id, first_ordinal, num_records, file_basename, written_records ]
        """
        if not isinstance(write_ready_inputs, (tuple, list)):
            write_ready_inputs = tuple(write_ready_inputs)

        # the buffers are already compressed into buffers in make_compress_stage()
        to_writer_gen = (
            (buffer_list_handle, record_id, first_ordinal, num_records, file_basename)
            for buffer_list_handle, num_records, first_ordinal, record_id, id_and_count, file_basename in write_ready_inputs
        )
        around_writer_gen = (
            (id_and_count, record_id, first_ordinal, num_records, file_basename)
            for buffer_list_handle, num_records, first_ordinal, record_id, id_and_count, file_basename in write_ready_inputs
        )

        written_records = (
            tuple(a) for a in pipeline.local_write_pipeline(upstream_tensors=to_writer_gen,
                                                            compressed=(self.compress_parallel>0),
                                                            record_types=self.write_columns)
        )

        final_output_gen = (a+b for a,b in zip(around_writer_gen, written_records))
        return final_output_gen

    def make_graph_impl(self, local_gate):
        """
        :param local_gate:
        :param args:
        :return: a gen of [ id_and_count, record_id, first_ordinal, num_records, file_basename, written_records]
        """
        with tf.name_scope("read_stage"):
            # read ops: [ id_and_count, [ filename ], [ a list of handles in the order of the columns, NOT STACKED ] ]
            read_ops = tuple(self.make_read_stage(gate=local_gate))
        # same as read ops, but flattened for ease of queueing
        read_ops_flattened = tuple((a[0],)+tuple(a[2:])+tuple(a[1]) for a in read_ops)
        write_ready_inputs = self.make_central_pipeline(inputs=read_ops_flattened)

        with tf.name_scope("write_stage"):
            write_ops = self.make_write_stage(write_ready_inputs=write_ready_inputs)

        queue_name = "written_records"
        all_done = pipeline.join(upstream_tensors=write_ops,
                                 parallel=1, multi=True,
                                 capacity=self.final_sink_capacity,
                                 name=queue_name, shared_name=queue_name)
        assert len(all_done) == 1
        return all_done[0]

class CephSnapStage(SnapCommonStage, Ceph):

    @staticmethod
    def type_name():
        return "ceph"

    @classmethod
    def add_graph_args(cls, parser):
        cls.add_common_graph_args(parser=parser)
        cls.add_ceph_args(parser=parser)

    def __init__(self, args):
        super().__init__(args)
        self.add_ceph_attrs(args=args)

    def make_read_stage(self, gate):
        """
        :param gate: each gate has components (key, namespace) for each chunk file to read (as the basename)
        :param args:
        :return: a generator of [ id_and_count, (key, namespace, [ unstacked list of handles ]) ]
        """
        # each item in dequeue_ops' components is a single filename
        dequeue_ops = tuple(gate.dequeue() for _ in range(self.read_parallel))
        ids_and_counts = tuple(d.id_and_count for d in dequeue_ops)
        kwargs = {}
        if self.log_goodput:
            kwargs["log_directory"] = self.log_directory
            kwargs["metadata"] = tuple(slice_id(idc) for idc in ids_and_counts)

        # comp_gen: key, namespace, [chunk_buffers_for_column]
        comp_gen = pipeline.ceph_read_pipeline(
            upstream_tensors=(tf.unstack(d.components[0], name="ceph_read_unstack") for d in dequeue_ops),
            user_name=self.ceph_user_name,
            cluster_name=self.ceph_cluster_name,
            ceph_conf_path=str(self.ceph_conf_path.absolute()),
            ceph_read_size=self.ceph_read_chunk_size,
            pool_name=self.ceph_pool_name,
            columns=self.columns,
            name="align_ceph_read",
            **kwargs
        )

        return zip((d.id_and_count for d in dequeue_ops), comp_gen)

    def make_write_stage(self, write_ready_inputs):
        """
        Note: compressed results column is already in buffer format and compressed
        :param args:
        :param write_ready_inputs: a generator of [ compressed_results_column_matrix, num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        :return: a gen of [ id_and_count, record_id, first_ordinal, num_records, file_basename, written_records ]
        """
        if not isinstance(write_ready_inputs, (tuple, list)):
            write_ready_inputs = tuple(write_ready_inputs)

        to_writer_gen = (
            (key, namespace, num_records, first_ordinal, record_id, buffer_handle)
            for buffer_handle, num_records, first_ordinal, record_id, id_and_count, key, namespace in write_ready_inputs
        )
        around_writer_gen = tuple(
            (id_and_count, record_id, first_ordinal, num_records, key, namespace)
            for buffer_handle, num_records, first_ordinal, record_id, id_and_count, key, namespace in write_ready_inputs
        )

        kwargs = {}
        if self.log_goodput:
            kwargs["log_directory"] = self.log_directory
            kwargs["metadata"] = tuple(slice_id(a[0]) for a in around_writer_gen)

        written_records = (
            tuple(a) for a in pipeline.ceph_write_pipeline(
                upstream_tensors=to_writer_gen,
                user_name=self.ceph_user_name,
                cluster_name=self.ceph_cluster_name,
                pool_name=self.ceph_pool_name,
                ceph_conf_path=str(self.ceph_conf_path),
                compressed=True,
                record_types=self.write_columns,
                name="align_ceph_write",
                **kwargs
            )
        )

        final_output_gen = (a+b for a,b in zip(around_writer_gen, written_records))
        return final_output_gen

    def make_graph_impl(self, local_gate):
        """
        :param local_gate:
        :param args:
        :return: a gen of [ id_and_count, record_id, first_ordinal, num_records, key, namespace, written_records]
        """
        with tf.name_scope("read_stage"):
            # read ops: a generator of [ id_and_count, (key, namespace, [ unstacked list of handles ]) ]
            read_ops = tuple(self.make_read_stage(gate=local_gate)) # tuple so that they're made in scope
        # same as read ops, but flattened for ease of queueing
        column_boundary = len(self.columns)
        read_ops_flattened = tuple((idc,)+tuple(comp[column_boundary])+tuple(comp[:column_boundary]) for idc, comp in read_ops)
        write_ready_inputs = self.make_central_pipeline(inputs=read_ops_flattened)

        with tf.name_scope("write_stage"):
            write_ops = self.make_write_stage(write_ready_inputs=write_ready_inputs)

        queue_name = "written_records"
        all_done = pipeline.join(upstream_tensors=write_ops,
                                 parallel=1, multi=True,
                                 capacity=self.final_sink_capacity,
                                 name=queue_name, shared_name=queue_name)

        assert len(all_done) == 1
        return all_done[0]
