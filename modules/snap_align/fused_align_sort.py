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
from functools import partial
from .common import *
import itertools

import tensorflow as tf
from abc import abstractmethod

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import pipeline
import tensorflow.contrib.gate as gate

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)

location_value = "location"

class FusedCommonStage(Stage):
    columns = (base_extension, qual_extension, metadata_extension)
    local_dest = "fused"

    def __init__(self, args):
        super().__init__()
        self._run_first = []
        expected_args = ("read_parallel", "decompress_parallel", "align_parallel", "aligner_threads", "sort_parallel", "sink_parallel",
                         "write_parallel", "deep_verify", "paired", "snap_args", "subchunking", "index_path", "max_secondary",
                         "sort_batch", "order_by", "base_pack_parallel", "log_goodput", "log_directory",
                         "aligner_thread_expansion_factor")
        for expected_arg in expected_args:
            arg_name = "_".join((self.local_dest, expected_arg))
            setattr(self, expected_arg, getattr(args, arg_name))

        if hasattr(args, "log_goodput") and args.log_goodput is True:
            self.log.info("Override enabling log goodput from global param")
            self.log_goodput = args.log_goodput

        queue_length_defaults = (
            ("head_gate_capacity", "sort_parallel", 0, 2), # in terms of number of batches
            ("pre_decomp_capacity", "read_parallel", 1, 0),
            ("pre_align_capacity", "align_parallel", 1, 0),
            ("pre_write_capacity", "base_pack_parallel", 1, 0),
            ("final_sink_capacity", "sink_parallel", 4, 0), # this queue should basically be empty for the most part
            ("pre_sort_gate_capacity", "sort_parallel", 1, 2),
            ("base_pack_capacity", "align_parallel", 1, 0),
        )

        for queue_cap_attr, source_attr, expansion_factor, additive_factor in queue_length_defaults:
            args_attr = "_".join((self.local_dest, queue_cap_attr))
            args_value = getattr(args, args_attr)
            default_value = int(getattr(self, source_attr) * expansion_factor) + int(additive_factor)
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
        cls.prefix_option(parser=parser, prefix=prefix, argument="read-parallel", default=2, type=numeric_min_checker(1, "must have >0 parallel read stages"), help="number of read stages to run in parallel")
        cls.prefix_option(parser=parser, prefix=prefix, argument="decompress-parallel", default=3, type=numeric_min_checker(1, "must have >0 parallel decomp stages"), help="number of decompress stages to run in parallel")
        cls.prefix_option(parser=parser, prefix=prefix, argument="align-parallel", default=4, type=numeric_min_checker(1, "must have >0 parallel align stages"), help="number of parallel align stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="aligner-threads", default=multiprocessing.cpu_count()-3, type=numeric_min_checker(1, "must have >0 parallel aligner threads"), help="number of aligner threads for shared aligner")
        cls.prefix_option(parser=parser, prefix=prefix, argument="aligner-thread-expansion-factor", default=2, type=numeric_min_checker(1, "must have >1 expansion factor"), help="how large to make the aligner queue")
        cls.prefix_option(parser=parser, prefix=prefix, argument="base-pack-parallel", default=2, type=numeric_min_checker(1, "must have >0 parallel base packing stages"), help="number of parallel base packing stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="write-parallel", default=2, type=numeric_min_checker(1, "must have >0 parallel write stages"), help="number of parallel write stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sort-parallel", default=2, type=numeric_min_checker(1, "must have >0 parallel sort stages"), help="number of parallel sort stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sink-parallel", type=numeric_min_checker(1, "must have >0 parallel return stages"), default=2, help="number of parallel stages to return at the end of this pipeline")
        cls.prefix_option(parser=parser, prefix=prefix, argument="deep-verify", default=False, action='store_true', help="verify record integrity")
        cls.prefix_option(parser=parser, prefix=prefix, argument="paired", default=False, action='store_true', help="interpret dataset as interleaved paired dataset")
        cls.prefix_option(parser=parser, prefix=prefix, argument="snap-args", default="", help="SNAP algorithm specific self. Pass with enclosing \" \". E.g. \"-om 5 -omax 1\" . See SNAP documentation for all options.")
        cls.prefix_option(parser=parser, prefix=prefix, argument="subchunking", default=100, type=numeric_min_checker(50, "don't go lower than 100 for subchunking size"), help="the size of each subchunk (in number of reads)")
        # Note: can't have path-exists checker for this because the path might be on a remote machine
        cls.prefix_option(parser=parser, prefix=prefix, argument="index-path", default="/home/whitlock/tf/ref_index", help="location of the ref index on all machines. Make sure all machines have this path!")
        cls.prefix_option(parser=parser, prefix=prefix, argument="max-secondary", type=numeric_min_checker(0, "must have a non-negative number of secondary results"), default=0, help="Max secondary results to store. >= 0 ")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sort-batch", type=numeric_min_checker(1, "must have >=1 batch from global gate"), default=10, help="number of sorted AGD rows (file columns) to sort at a given time. the arity of the sorters")
        cls.prefix_option(parser=parser, prefix=prefix, argument="order-by", default=location_value, choices=(location_value, "metadata"), help="sort by this parameter [location | metadata]")

        # all options below here are rather verbose, for length of queues
        cls.prefix_option(parser=parser, prefix=prefix, argument="head-gate-capacity", type=numeric_min_checker(1, "must have at least one head request"), help="")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-decomp-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-read, pre-decomp queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-align-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-decomp, pre-align queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="base-pack-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-align, pre-base packing queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-sort-gate-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity of the gate in each align/sort stage, right before sorting")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-write-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-align, pre-write queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="final-sink-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity of final queue of this stage")

        cls.prefix_option(parser=parser, prefix=prefix, argument="log-goodput", default=False, action="store_true", help="log the goodput events")
        cls.prefix_option(parser=parser, prefix=prefix, argument="log-directory", default="/home/whitlock/tf/shell", help="the directory to log all events to, if log_goodput is enabled")

    def make_central_pipeline(self, inputs, local_head_gate):
        """
        Make the central pipeline between the custom read and write operations
        :param args:
        :param inputs: a generator of type (id_and_count, column0, column1, ..., [:rest of input]). The number of colums is assumed to be the same and in the same order as self.columns
        :return: a generator of [ id_and_count, record_id, intermediate_name, num_recs, superchunk_matrix, {rest of input} ]
        """

        if not isinstance(inputs, (list, tuple)):
            inputs = tuple(inputs)

        # type of each of these: (id_and_count, column0, column1, ..., [:rest of input])
        queue_name = "ready_to_decomp"
        ready_to_decomp = pipeline.join(upstream_tensors=inputs,
                                        parallel=self.decompress_parallel,
                                        capacity=self.pre_decomp_capacity, multi=True,
                                        name=queue_name, shared_name=queue_name)

        with tf.name_scope("decompression_stage"):
            ready_to_align_items = tuple(self.make_decomp_stage(ready_to_decomp=ready_to_decomp))

        queue_name = "ready_to_align"
        ready_to_align = pipeline.join(upstream_tensors=ready_to_align_items,
                                       parallel=self.align_parallel,
                                       capacity=self.pre_align_capacity, multi=True,
                                       name=queue_name, shared_name=queue_name)

        with tf.name_scope("align_stage"):
            aligned_result_items = tuple(self.make_align_stage(ready_to_align=ready_to_align)) # we have to iterate over this multiple times
        # aligned_results_items: a tuple of [ results_column_matrix, [output_buffer_handles], num_records, first_ordinal, record_id, id_and_count, {rest of input} ]

        queue_name = "ready_to_base_pack"
        ready_to_convert_bases = pipeline.join(upstream_tensors=aligned_result_items,
                                               parallel=self.base_pack_parallel,
                                               capacity=self.base_pack_capacity, multi=True,
                                               name=queue_name, shared_name=queue_name)

        with tf.name_scope("base_packing_stage"):
            output_buffer_groups = tuple(self.make_base_packing_stage(output_buffer_groups=((a[1], a[2]) for a in ready_to_convert_bases)))

        converted_results = tuple((a[0],output_buffer_group)+tuple(a[2:])
                                  for a, output_buffer_group in zip(ready_to_convert_bases, output_buffer_groups))
        # converted results type:
        # results_column_matrix, [matrix of buffers], num_records, first_ordinal, record_id, id_and_count, {rest of input}

        id_and_count_index = 5
        ids_and_counts = tuple(a[id_and_count_index] for a in converted_results)
        assert len(ids_and_counts) > 0
        components = tuple(
            tuple(value for index, value in enumerate(aligned_result_item) if index != id_and_count_index)
            for aligned_result_item in converted_results
        )
        assert len(components) > 0
        assert len(components) == len(ids_and_counts)

        queue_name = "pre_sort_gate"
        pre_sort_gate = gate.StreamingGate(
            name=queue_name, shared_name=queue_name,
            id_and_count_upstream=ids_and_counts[0],
            sample_tensors=components[0],
            capacity=self.pre_sort_gate_capacity,
            limit_upstream=True, limit_downstream=False
        )
        gate.add_credit_supplier_from_gates(
            upstream_gate=local_head_gate,
            downstream_gate=pre_sort_gate
        )

        def gen_align_tail_timestamps():
            for _, _, num_records, first_ordinal, record_id, id_and_count in (a[:6] for a in converted_results):
                with tf.control_dependencies((id_and_count,)):
                    ts = gate.unix_timestamp(name="align_tail_timestamp")
                yield (gate.log_events(
                    item_names=("id", "time", "record_id", "num_records", "ordinal"),
                    directory=self.log_directory,
                    event_name="align_tail",
                    name="align_tail_event_logger",
                    components=(slice_id(id_and_count), ts, record_id, num_records, first_ordinal)
                ),)

        if self.log_goodput:
            control_deps = tuple(gen_align_tail_timestamps())
        else:
            control_deps = tuple(itertools.repeat((), times=len(ids_and_counts)))
        assert len(control_deps) == len(components)

        def gen_enqueue_ops():
            for idc, comp, control_dep in zip(ids_and_counts, components, control_deps):
                with tf.control_dependencies(control_dep):
                    yield pre_sort_gate.enqueue(
                        id_and_count=idc,
                        components=comp
                    )

        enqueue_ops = tuple(gen_enqueue_ops())
        assert len(enqueue_ops) == len(ids_and_counts) and len(enqueue_ops) == len(converted_results)
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=pre_sort_gate, enqueue_ops=enqueue_ops))

        dequeue_ops = tuple(pre_sort_gate.dequeue_many(count=self.sort_batch) for _ in range(self.sort_parallel))

        with tf.name_scope("sort_stage"):
            sorted = tuple(self.make_sort_stage(ready_to_sort=dequeue_ops))

        sorted_chunks, control_deps = zip(*sorted)

        queue_name = "ready_to_write"
        ready_to_write = pipeline.join(upstream_tensors=sorted_chunks,
                                       control_dependencies=control_deps,
                                       parallel=self.write_parallel,
                                       multi=True,
                                       capacity=self.pre_write_capacity,
                                       name=queue_name, shared_name=queue_name)

        return ready_to_write

    @property
    def column_extensions_for_write(self):
        secondary_results = tuple("".join((secondary_results_extension, str(i))) for i in range(self.max_secondary))
        if self.order_by == location_value:
            return (results_extension,) + tuple(self.columns) + secondary_results
        else:
            return (metadata_extension,) + tuple(c for c in self.columns if c != metadata_extension) + (results_extension,) + secondary_results

    def make_base_packing_stage(self, output_buffer_groups):
        """
        the output buffers, with the converted packed stuff
        :param output_buffer_groups: a tuple of the output buffers and the number of records
        :return:
        """
        converter = persona_ops.base_buffer_converter
        def convert_output_buffers(output_buffers, num_records):
            assert len(output_buffers) == len(self.columns)
            for column_extension, output_buffer in zip(self.columns, output_buffers):
                if column_extension == base_extension:
                    yield converter(num_records=num_records,
                                    buffer=output_buffer)
                else:
                    yield output_buffer
        for output_buffers, num_records in output_buffer_groups:
            unstacked = tf.unstack(output_buffers)
            yield tf.stack(tuple(convert_output_buffers(output_buffers=unstacked,
                                                        num_records=num_records)))

    def make_sort_stage(self, ready_to_sort):
        """
        Ready to sort is a batch dequeue, so it is striped in the first dimension of each
        :param ready_to_sort: a generator of id_and_count [ results_column_matrix, [output_buffer_handles], num_records, first_ordinal, record_id, {rest of input} ]
        :return: a generator of [ id_and_count, record_id, intermediate_filename_base, num_recs, superchunk_matrix, {rest of input} ], control_dependencies for pre-write
        """
        bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_sort_buffer_list_pool")

        for id_and_count, components in ready_to_sort:
            results_column_matrices, output_buffer_handless, num_recordss, first_ordinals, record_ids = components[:5]
            record_id = record_ids[0]
            first_ordinal = first_ordinals[0]
            rests_of_inputs = components[5:]
            rest_of_input = tuple(a[0] for a in rests_of_inputs)

            control_deps = []
            if self.log_goodput:
                with tf.control_dependencies((id_and_count,)):
                    ts = gate.unix_timestamp(name="sort_head_timestamp")
                control_deps.append(gate.log_events(
                    item_names=("id", "time"),
                    event_name="sort_head",
                    name="sort_head_event_logger",
                    directory=self.log_directory,
                    components=(slice_id(id_and_count), ts)
                ))


            first_ordinal_str = tf.as_string(first_ordinal, name="first_ordinal_conversion")

            # this filename is guaranteed to be unique because of the ordinal (unique among this dataset) and the extension (so it doesn't conflict with existing chunk files)
            # otherwise when a request is resubmitted, the cleanup from the merge stage may overlap with the new files created!
            random_gen = tf.as_string(
                tf.random_uniform(dtype=tf.int32, maxval=2**20, shape=(), name="random_intermediate_name_gen"),
                name="random_intermediate_value_to_string"
            )
            intermediate_name = tf.string_join((record_id, first_ordinal_str, random_gen, intermediate_extension), separator="_", name="intermediate_filename")

            if self.order_by == location_value:
                results_unstack = tf.unstack(results_column_matrices, axis=1, name="results_unstack") # [ results, secondary0, secondary1, ... ]
                results_column = results_unstack[0]
                secondary_results = results_unstack[1:]
                if len(secondary_results) > 0:
                    secondary_results = tf.stack(secondary_results, axis=1, name="secondary_results_stack")
                    output_buffer_handless = tf.concat((output_buffer_handless, secondary_results), axis=1, name="secondary_results_append")

                with tf.control_dependencies(control_deps):
                    superchunk_matrix, superchunk_num_recs = persona_ops.agd_sort(
                        buffer_pair_pool=bpp,
                        num_records=num_recordss,
                        column_handles=output_buffer_handless,
                        sort_key_handles=results_column,
                        name="agd_sort_results"
                    )
            else:
                raise Exception("Not supported!")
                metadata_column_index = self.columns.index(metadata_extension)
                unstacked_columns = tf.unstack(output_buffer_handless, axis=1, name="unstack_non_results_columns")
                metadata_column = unstacked_columns[metadata_column_index]
                other_columns = [ v for i,v in enumerate(unstacked_columns) if i != metadata_column_index]
                other_columns_stacked = tf.stack(other_columns, axis=1, name="stack_other_columns")
                all_other_columns = tf.concat((other_columns_stacked, results_column_matrices), axis=1, name="all_other_columns_concat")

                with tf.control_dependencies(control_deps):
                    superchunk_matrix, superchunk_num_recs = persona_ops.agd_sort_metadata(
                        buffer_pair_pool=bpp,
                        num_records=num_recordss,
                        column_handles=all_other_columns,
                        sort_key_handles=metadata_column,
                        name="agd_sort_metadata"
                    )

            if self.log_goodput:
                with tf.control_dependencies((superchunk_num_recs,)):
                    ts = gate.unix_timestamp(name="sort_tail_timestamp")
                log_event = (gate.log_events(
                    item_names=("id", "time", "record_id", "num_records"),
                    directory=self.log_directory,
                    event_name="sort_tail",
                    name="sort_tail_event_logger",
                    components=(slice_id(id_and_count), ts, record_id, superchunk_num_recs)
                ),)
            else:
                log_event = ()

            yield (id_and_count, record_id, intermediate_name, superchunk_num_recs, superchunk_matrix) + rest_of_input, log_event

    def make_align_stage(self, ready_to_align):
        """
        :param ready_to_align: a generator of [ agd_read_handle, [output_buffer_handles], num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        :return: a generator of [ results_column_matrix, [output_buffer_handles], num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        """
        ready_to_align = sanitize_generator(ready_to_align)
        to_aligners, around_aligners = zip(*(
            (a[0], a[1:]) for a in ready_to_align
        ))

        def gen_timestamps():
            for group in around_aligners:
                with tf.control_dependencies((group[4],)):
                    yield gate.unix_timestamp(name="pure_align_head_timestamp")

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
                                        work_queue_size=self.aligner_threads*self.aligner_thread_expansion_factor,
                                        options_handle=aligner_options,
                                        genome_handle=genome)

        aligner_control_deps = itertools.repeat((), times=len(to_aligners))
        timestamps = itertools.repeat((), times=len(to_aligners))
        if False and self.log_goodput:
            timestamps = tuple((a,) for a in gen_timestamps())
            aligner_control_deps = [
                (gate.log_events(
                    item_names=("id", "time", "ordinal"),
                    event_name="pure_align_head",
                    name="pure_align_head_event_logger",
                    directory=self.log_directory,
                    components=(in_id, timestamp[0], ordinal)
                ),) for in_id, timestamp, ordinal in zip(
                    (slice_id(a[4]) for a in around_aligners),
                    timestamps,
                    (a[2] for a in around_aligners)
                )
            ]


        aligner_op = partial(aligner_type,
                             buffer_list_pool=buffer_list_pool,
                             subchunk_size=self.subchunking,
                             executor_handle=single_executor,
                             max_secondary=self.max_secondary,
                             release_resources=False)

        def coalesced_results():
            bp = persona_ops.buffer_pool(size=0, bound=False, name="buffer_list_to_buffer_converter_pool")
            converter_op = partial(persona_ops.buffer_list_to_buffer_converter,
                                   buffer_pool=bp)
            for read_handle, control_dep, timestamp_dep in zip(to_aligners, aligner_control_deps, timestamps):
                with tf.control_dependencies(timestamp_dep):
                    aligner_result = aligner_op(read=read_handle)
                individual_results = tf.unstack(aligner_result)
                converted_results = tuple(
                    converter_op(buffer_list=bl) for bl in individual_results
                )
                with tf.control_dependencies(control_dep):
                    yield tf.stack(converted_results)

        for aligner_result, around_aligner in zip(coalesced_results(), around_aligners):
            yield (aligner_result,) + tuple(around_aligner)

    def make_decomp_stage(self, ready_to_decomp):
        """
        :param ready_to_decomp: generator of (id_and_count, column0, column1, ..., [:rest of input])
        :return: a generator of [ agd_read_handle, [output_buffer_handles], num_records, first_ordinal, record_id, id_and_count, {rest of input} ]
        """
        ready_to_decomp = sanitize_generator(ready_to_decomp)

        num_needed_columns = len(self.columns) # need to decomp all columns, even metadata which we don't read

        #around_everything: (id_and_count, {rest, of, input, ...})
        # needs to be tuple in case of log goodput
        around_everything = tuple((a[0],) + tuple(a[1+num_needed_columns:]) for a in ready_to_decomp)

        # to_agd_reader = just the needed columns
        to_agd_reader = tuple((a[1:1+num_needed_columns] for a in ready_to_decomp))

        def gen_timestamps():
            for group in ready_to_decomp:
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
        multi_column_gen = tuple(pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_agd_reader, verify=self.deep_verify, name="align_sort_reader", **reader_kwargs))

        assembled_columns = len(self.columns) - 1 # don't need the metadata

        #  to_assembler: ([assembled_read_handles], num_records)
        to_assembler = tuple((a[0][:assembled_columns], a[1]) for a in multi_column_gen)

        assembler_kwargs = {}
        if self.log_goodput:
            log_event_ops = [
                (gate.log_events( # single element tuple because that's how tf.control_dependencies works
                    item_names=("id","time", "ordinal"),
                    components=(in_id, timestamp, ordinal),
                    event_name="align_head",
                    directory=self.log_directory,
                    name="align_head_event_logger"
                ),) for in_id, timestamp, ordinal in zip(
                    (slice_id(a[0]) for a in around_everything),
                    timestamps,
                    (a[2] for a in multi_column_gen)
                )
            ]
            assembler_kwargs["control_deps"] = log_event_ops

        # each element is an agd_reads handle
        agd_assembled_reads = pipeline.agd_read_assembler(upstream_tensors=to_assembler, include_meta=False, **assembler_kwargs)
        for agd_read_handle, around_assembler_group, around_everything_group in zip(agd_assembled_reads, multi_column_gen, around_everything):
            yield (agd_read_handle,) + tuple(around_assembler_group) + tuple(around_everything_group)

    @abstractmethod
    def make_graph_impl(self, local_gate):
        raise NotImplementedError

    def make_head_gate(self, upstream_gate):
        id_and_count, components = upstream_gate.dequeue_partition(count=self.sort_batch)
        gate_name = "head_gate"
        head_gate = gate.StreamingGate(limit_upstream=False, # we don't limit upstream, as that is from the central queue
                                       limit_downstream=True, # we do limit the downstream to the batching join gate so it doesn't suck up all resources
                                       id_and_count_upstream=id_and_count, sample_tensors=components,
                                       name=gate_name, shared_name=gate_name,
                                       capacity=self.head_gate_capacity,
                                       sample_tensors_are_batch=True)
        enq_ops = (head_gate.enqueue_many(id_and_count=id_and_count, components=components),)
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=head_gate, enqueue_ops=enq_ops))
        return head_gate

    def _make_graph(self, upstream_gate):
        head_gate = self.make_head_gate(upstream_gate=upstream_gate)
        return self.make_graph_impl(local_gate=head_gate)

class LocalFusedStage(FusedCommonStage):
    path_separator_str = "/"
    path_separator = tf.constant(path_separator_str)

    @classmethod
    def add_local_graph_args(cls, parser):
        """
        Adds graph arguments for reading and writing locally
        :param parser:
        :return:
        """
        cls.prefix_option(parser=parser, prefix=cls.local_dest, argument="path-prefix", default="", help="path prefix to assign to this fused_align_sort, for example for a common FUSE mount point")

    @classmethod
    def add_graph_args(cls, parser):
        cls.add_common_graph_args(parser=parser)
        cls.add_local_graph_args(parser=parser)

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
            if path_prefix[-1] != self.path_separator_str:
                path_prefix = path_prefix + self.path_separator_str
            path_prefix = tf.constant(path_prefix)
            filenames = (tf.string_join((path_prefix, fname)) for fname in filenames)
        read_file_gen = zip(dequeue_ops, pipeline.local_read_pipeline(
            upstream_tensors=filenames, columns=self.columns
        ))
        for a,b in read_file_gen:
            yield tuple(a)+tuple(b)

    def make_write_stage(self, write_ready_inputs):
        """
        :param write_ready_inputs: a generator of [ id_and_count, record_id, intermediate_filename_base, num_recs, superchunk_matrix, filename_base ]
        :return: a gen of [ id_and_count, full_path, num_records, [list of written paths, for each column] ]
        """
        write_ready_inputs = sanitize_generator(write_ready_inputs)

        extensions = self.column_extensions_for_write
        write_types = tuple(get_type_for_extension(column_extension=c, text_base=False) for c in extensions)
        for idc, record_id, intermediate_basename, num_recs, superchunk_matrix, directory in write_ready_inputs:
            assert superchunk_matrix.shape[0] == len(extensions)
            full_path = tf.string_join((directory, intermediate_basename), separator=self.path_separator_str, name="full_intermediate_chunkfile_prefix")
            chunk_files = tf.unstack(superchunk_matrix)
            assert len(chunk_files) == len(extensions)

            first_ordinal = tf.constant(0, dtype=tf.int64)
            writer = partial(persona_ops.agd_file_system_buffer_pair_writer,
                             first_ordinal=first_ordinal,
                             record_id=record_id,
                             num_records=num_recs)

            written_paths = []
            for chunk_file, extension, write_type in zip(chunk_files, extensions, write_types):
                chunk_full_path = tf.string_join([full_path, extension], separator=".", name="final_full_path_{}".format(extension))
                with tf.control_dependencies(written_paths):
                    result = writer(
                        record_type=write_type,
                        path=chunk_full_path,
                        resource_handle=chunk_file,
                        name="intermediate_writer_{ext}_{t}".format(ext=extension, t=write_type)
                    )
                    written_paths.append(result)

            # have to include the written paths, even though we discard them, so that the ops get triggered
            yield idc, full_path, num_recs, written_paths

    def make_graph_impl(self, local_gate):
        """
        :param local_gate:
        :param args:
        :return:
        """
        with tf.name_scope("read_stage"):
            # read ops: [ id_and_count, [ filename ], [ a list of handles in the order of the columns, NOT STACKED ] ]
            read_ops = (self.make_read_stage(gate=local_gate))
        # same as read ops, but flattened for ease of queueing
        read_ops_flattened = ((a[0],)+tuple(a[2:]+tuple(a[1])) for a in read_ops)
        ready_to_write_chunks = self.make_central_pipeline(inputs=read_ops_flattened, local_head_gate=local_gate)

        def gen_downstream():
            for ready_to_write_chunk in ready_to_write_chunks:
                processed = ready_to_write_chunk[:-1]
                chunkfile_basename = ready_to_write_chunk[-1]

                yield tuple(processed) + (dirname(chunkfile_basename),)

        with tf.name_scope("write_stage"):
            written_records = self.make_write_stage(write_ready_inputs=gen_downstream())

        queue_name = "completed_results"
        sink_queue = pipeline.join(upstream_tensors=written_records,
                                   parallel=self.sink_parallel,
                                   multi=True,
                                   capacity=self.final_sink_capacity,
                                   name=queue_name,
                                   shared_name=queue_name)
        return tuple(s[:-1] for s in sink_queue) # :-1 to leave off the file records that aren't needed

class CephFusedStage(FusedCommonStage, Ceph):

    @classmethod
    def add_local_graph_args(cls, parser):
        """
        Adds graph arguments for reading and writing locally
        :param parser:
        :return:
        """
        cls.add_ceph_args(parser=parser)

    @classmethod
    def add_graph_args(cls, parser):
        cls.add_common_graph_args(parser=parser)
        cls.add_local_graph_args(parser=parser)

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

        kwargs = {}
        if self.log_goodput:
            kwargs["log_directory"] = self.log_directory
            kwargs["metadata"] = tuple(slice_id(d.id_and_count) for d in dequeue_ops)

        # comp_gen: key, namespace, [chunk_buffers_for_column]
        comp_gen = pipeline.ceph_read_pipeline(
            upstream_tensors=(tf.unstack(d.components[0], name="ceph_read_unstack") for d in dequeue_ops),
            user_name=self.ceph_user_name,
            cluster_name=self.ceph_cluster_name,
            ceph_conf_path=str(self.ceph_conf_path.absolute()),
            ceph_read_size=self.ceph_read_chunk_size,
            pool_name=self.ceph_pool_name,
            columns=self.columns,
            name="fused_ceph_read",
            **kwargs
        )

        return zip((d.id_and_count for d in dequeue_ops), comp_gen)

    def make_write_stage(self, write_ready_inputs):
        """
        :param write_ready_inputs: a generator of [ id_and_count, record_id, intermediate_name, num_recs, superchunk_matrix, {rest of input} ]
        :return: a gen of [ id_and_count, intermediate_name, namespace, num_recs, [list of written paths, for each column] ]
        """
        if not isinstance(write_ready_inputs, (tuple, list)):
            write_ready_inputs = tuple(write_ready_inputs)

        extensions = self.column_extensions_for_write
        write_types = tuple(get_type_for_extension(column_extension=c, text_base=False) for c in extensions)
        for idc, record_id, intermediate_name, num_recs, superchunk_matrix, key, namespace in write_ready_inputs:
            assert superchunk_matrix.shape[0] == len(extensions)
            chunk_files = tf.unstack(superchunk_matrix)
            assert len(chunk_files) == len(extensions)

            first_ordinal = tf.constant(0, dtype=tf.int64)
            writer = partial(persona_ops.agd_ceph_buffer_pair_writer,
                             first_ordinal=first_ordinal,
                             record_id=record_id,
                             namespace=namespace,
                             cluster_name=self.ceph_cluster_name,
                             user_name=self.ceph_user_name,
                             ceph_conf_path=str(self.ceph_conf_path),
                             pool_name=self.ceph_pool_name,
                             num_records=num_recs)

            def gen_written_paths():
                for chunk_file, extension, write_type in zip(chunk_files, extensions, write_types):
                    full_key = tf.string_join([intermediate_name, extension], separator=".", name="full_key_join_{ext}_{t}".format(
                        ext=extension, t=write_type
                    ))
                    a = writer(
                        path=full_key,
                        record_type=write_type,
                        resource_handle=chunk_file,
                        name="intermediate_ceph_writer_{ext}_{t}".format(ext=extension, t=write_type)
                    )
                    out_path = a.output_path
                    if self.log_goodput:
                        timestamp = a.time
                        write_duration = a.duration
                        write_size = a.bytes
                        log_op = gate.log_events(
                            item_names=("timestamp", "key", "duration", "bytes", "id"),
                            directory=self.log_directory,
                            event_name="fused_ceph_write",
                            name="fused_ceph_write",
                            components=(timestamp, out_path, write_duration, write_size, slice_id(idc))
                        )
                        with tf.control_dependencies((log_op,)):
                            out_path = tf.identity(out_path)
                    yield out_path

            # have to include the written paths, even though we discard them, so that the ops get triggered
            yield idc, intermediate_name, namespace, num_recs, tuple(gen_written_paths())

    def make_graph_impl(self, local_gate):
        """
        :param local_gate:
        :param args:
        :return: a single item of [ id_and_count, intermediate_name, namespace ]
        """
        with tf.name_scope("read_stage"):
            # read ops: a generator of [ id_and_count, (key, namespace, [ unstacked list of handles ]) ]
            read_ops = tuple(self.make_read_stage(gate=local_gate)) # tuple so that they're made in scope
        # same as read ops, but flattened for ease of queueing
        column_boundary = len(self.columns)-1
        assert column_boundary >= 2
        read_ops_flattened = tuple((idc,)+tuple(comp[column_boundary])+tuple(comp[:column_boundary]) for idc, comp in read_ops)

        # :return: a generator of [ id_and_count, record_id, intermediate_name, num_recs, superchunk_matrix, {rest of input} ]
        # where {rest of input} = key, namespace
        ready_to_write = self.make_central_pipeline(inputs=read_ops_flattened, local_head_gate=local_gate)

        with tf.name_scope("write_stage"):
            written_records = self.make_write_stage(write_ready_inputs=ready_to_write)

        queue_name = "completed"
        sink_queue = pipeline.join(upstream_tensors=written_records,
                                   parallel=self.sink_parallel,
                                   multi=True,
                                   capacity=self.final_sink_capacity,
                                   name=queue_name,
                                   shared_name=queue_name)
        return tuple(s[:-1] for s in sink_queue) # :-1 to leave off the file records that aren't needed

class SmallFusedCommonStage(FusedCommonStage, ShareArguments):
    local_dest = "{}small".format(FusedCommonStage.local_dest)

    default_parallelism_args = (
        ("read_parallel", 3),
        ("decompress_parallel", 3),
        ("align_parallel", 6),
        ("aligner_threads", max((multiprocessing.cpu_count()/2)+6, 1)),
        ("write_parallel", 3),
        ("base_pack_parallel", 2),
        ("sort_parallel", 3)
    )

    @classmethod
    def add_common_graph_args(cls, parser):
        super().add_common_graph_args(parser=parser)
        new_args = {
            "{ld}_{k}".format(ld=cls.local_dest, k=k): v
            for k,v in cls.default_parallelism_args
        }
        assert all(parser.get_default(n) is not None for n in new_args.keys()), "Incorrectly specified defaults: [ {} ]".format(", ".join(n for n in new_args.keys() if parser.get_default(n) is None))
        parser.set_defaults(**new_args)

    args_from_fused_if_exist = (
        "paired", "snap_args", "max_secondary", "order_by", "deep_verify"
    )

    def __init__(self, args):
        super().__init__(args=args)
        self.get_arguments_from_other(attrs=self.args_from_fused_if_exist, args=args, other_dest=FusedCommonStage.local_dest, log=self.log,
                                      this_dest=self.local_dest)

class SmallCephFusedStage(SmallFusedCommonStage, CephFusedStage):
    def __init__(self, args):
        super().__init__(args=args)
        self.get_arguments_from_other(
            attrs=self.ceph_attributes, args=args, other_dest=CephFusedStage.local_dest,
            log=self.log, this_dest=self.local_dest
        )

class SmallLocalFusedStage(SmallFusedCommonStage, LocalFusedStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
