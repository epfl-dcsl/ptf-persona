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
from ..common.stage import Stage
from functools import partial
from .common import *
from .fused_align_sort import location_value
from .snap_align import SnapCommonStage
import itertools
from abc import abstractmethod

import tensorflow as tf
persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib import gate
from tensorflow.contrib.persona import pipeline

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)

class SortCommonStage(Stage, ShareArguments):
    local_dest = "sort"
    columns = (base_extension, qual_extension, metadata_extension)

    @property
    def extended_columns(self):
        secondary_results = tuple("".join((secondary_results_extension, str(i))) for i in range(self.max_secondary))
        if self.order_by == location_value:
            return (results_extension,) + tuple(self.columns) + secondary_results
        else:
            return (metadata_extension,) + tuple(c for c in self.columns if c != metadata_extension) + (results_extension,) + secondary_results

    def __init__(self, args):
        super().__init__()
        expected_args = ("read_parallel", "decompress_parallel", "sort_parallel", "sink_parallel",
                         "write_parallel", "deep_verify", "max_secondary",
                         "sort_batch", "order_by", "log_goodput", "log_directory")
        for expected_arg in expected_args:
            arg_name = "_".join((self.local_dest, expected_arg))
            setattr(self, expected_arg, getattr(args, arg_name))

        if hasattr(args, "log_goodput") and args.log_goodput is True:
            self.log.info("Override enabling log goodput from global param")
            self.log_goodput = args.log_goodput

        queue_length_defaults = (
            ("head_gate_capacity", "sort_parallel", 0, 2), # in terms of number of batches
            ("pre_decomp_capacity", "read_parallel", 1, 0),
            ("pre_sort_gate_capacity", "sort_parallel", 0.5, 1),
            ("pre_write_capacity", "write_parallel", 0.5, 1),
            ("final_sink_capacity", "write_parallel", 1, 1), # this queue should basically be empty for the most part
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
        self.get_arguments_from_other(
            attrs=("max_secondary",), this_dest=self.local_dest,
            other_dest=SnapCommonStage.local_dest, args=args, log=self.log
        )

    @classmethod
    def add_graph_args(cls, parser):
        prefix=cls.local_dest
        cls.prefix_option(parser=parser, prefix=prefix, argument="read-parallel", default=6, type=numeric_min_checker(1, "must have >0 parallel read stages"), help="number of read stages to run in parallel")
        cls.prefix_option(parser=parser, prefix=prefix, argument="decompress-parallel", default=16, type=numeric_min_checker(1, "must have >0 parallel decomp stages"), help="number of decompress stages to run in parallel")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sort-parallel", default=6, type=numeric_min_checker(1, "must have >0 parallel sort stages"), help="number of parallel sort stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="write-parallel", default=6, type=numeric_min_checker(1, "must have >0 parallel write stages"), help="number of parallel write stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sink-parallel", default=2, type=numeric_min_checker(1, "must have >0 parallel return stages"), help="number of parallel stages to return at the end of this pipeline")
        cls.prefix_option(parser=parser, prefix=prefix, argument="deep-verify", default=False, action='store_true', help="verify record integrity")
        # Note: can't have path-exists checker for this because the path might be on a remote machine
        cls.prefix_option(parser=parser, prefix=prefix, argument="sort-batch", type=numeric_min_checker(1, "must have >=1 batch from global gate"), default=10, help="number of sorted AGD rows (file columns) to sort at a given time. the arity of the sorters")
        cls.prefix_option(parser=parser, prefix=prefix, argument="order-by", default=location_value, choices=(location_value, "metadata"), help="sort by this parameter [location | metadata]")

        # left blank so we can get this from align
        cls.prefix_option(parser=parser, prefix=prefix, argument="max-secondary", type=numeric_min_checker(0, "must have a non-negative number of secondary results"), help="Max secondary results to store. >= 0 ")

        # all options below here are rather verbose, for length of queues
        cls.prefix_option(parser=parser, prefix=prefix, argument="head-gate-capacity", type=numeric_min_checker(1, "must have at least one head request"), help="")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-decomp-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-read, pre-decomp queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-sort-gate-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity of the gate in each align/sort stage, right before sorting")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-write-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-align, pre-write queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="final-sink-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity of final queue of this stage")

        cls.prefix_option(parser=parser, prefix=prefix, argument="log-goodput", default=False, action="store_true", help="log the goodput events")
        cls.prefix_option(parser=parser, prefix=prefix, argument="log-directory", default="/home/whitlock/tf/shell", help="the directory to log all events to, if log_goodput is enabled")

    def make_central_pipeline(self, inputs, local_head_gate):
        """
        :param inputs:
        :param local_head_gate:
        :return: (id_and_count, record_id, intermediate_name, superchunk_num_records, superchunk_matrix) + rest_of_input
        """
        inputs = sanitize_generator(inputs)
        queue_name = "sort_ready_to_decomp"
        ready_to_decomp = pipeline.join(upstream_tensors=inputs,
                                        parallel=self.decompress_parallel,
                                        capacity=self.pre_decomp_capacity, multi=True,
                                        name=queue_name, shared_name=queue_name)
        with tf.name_scope("decompression_stage"):
            ready_to_sort_items = sanitize_generator(self.make_decomp_stage(ready_to_decomp=ready_to_decomp))
        assert len(ready_to_sort_items) > 0

        queue_name = "pre_sort_gate"
        example_item = ready_to_sort_items[0]
        pre_sort_gate = gate.StreamingGate(
            name=queue_name, shared_name=queue_name,
            id_and_count_upstream=example_item[0],
            sample_tensors=example_item[1:],
            capacity=self.pre_sort_gate_capacity,
            limit_upstream=True, limit_downstream=False
        )
        gate.add_credit_supplier_from_gates(
            upstream_gate=local_head_gate,
            downstream_gate=pre_sort_gate
        )

        enqueue_ops = tuple(
            pre_sort_gate.enqueue(id_and_count=a[0],
                                  components=a[1:])
            for a in ready_to_sort_items
        )
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=pre_sort_gate, enqueue_ops=enqueue_ops))

        to_sort_ops = tuple(pre_sort_gate.dequeue_many(count=self.sort_batch) for _ in range(self.sort_parallel))

        with tf.name_scope("sort_stage"):
            sorted = tuple(self.make_sort_stage(ready_to_sort=to_sort_ops))
        sorted_chunks, control_deps = zip(*sorted)

        queue_name = "sort_ready_to_write"
        ready_to_write = pipeline.join(upstream_tensors=sorted_chunks,
                                       control_dependencies=control_deps,
                                       parallel=self.write_parallel,
                                       multi=True,
                                       capacity=self.pre_write_capacity,
                                       name=queue_name, shared_name=queue_name)

        return ready_to_write

    def make_sort_stage(self, ready_to_sort):
        """
        :param ready_to_sort:
        :return: (id_and_count, record_id, intermediate_name, superchunk_num_records, superchunk_matrix) + rest_of_input, log_event
        """
        bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="local_read_sort_buffer_list_pool")

        self.log.info("order by is '{ob}'".format(ob=self.order_by))
        if self.order_by == location_value:
            self.log.info("sorting by location")
            sort_op = partial(persona_ops.agd_sort,
                              buffer_pair_pool=bpp,
                              name="agd_sort_results")
        else:
            raise Exception("not supported")
            sort_op = partial(persona_ops.agd_sort_metadata,
                              buffer_pair_pool=bpp,
                              name="agd_sort_metadata")

        for id_and_count, components in ready_to_sort:
            output_buffer_handless, num_recordss, first_ordinals, record_ids = components[:4]
            rest_of_inputs = components[4:]

            # need to just pick the top things
            rest_of_input = tuple(a[0] for a in rest_of_inputs)
            record_id = record_ids[0]
            first_ordinal = first_ordinals[0]

            first_ordinal_str = tf.as_string(first_ordinal, name="first_ordinal_conversion")

            # this filename is guaranteed to be unique because of the ordinal (unique among this dataset) and the extension (so it doesn't conflict with existing chunk files)
            # otherwise when a request is resubmitted, the cleanup from the merge stage may overlap with the new files created!
            random_gen = tf.as_string(
                tf.random_uniform(dtype=tf.int32, maxval=2**20, shape=(), name="random_intermediate_name_gen"),
                name="random_intermediate_value_to_string"
            )
            intermediate_name = tf.string_join((record_id, first_ordinal_str, random_gen, intermediate_extension), separator="_", name="intermediate_filename")

            # TODO not sure if this axis=1 is correct
            unstack_handles = tf.unstack(output_buffer_handless, axis=1, name="buffers_unstack")
            key_handles = unstack_handles[0] # output_buffer_handless[:,0,:]
            other_handles = tf.stack(unstack_handles[1:], axis=1) # output_buffer_handless[:,1:,:]

            # first column is always the correct one, due to self.extended_columns order
            superchunk_matrix, superchunk_num_records = sort_op(
                num_records=num_recordss,
                sort_key_handles=key_handles,
                column_handles=other_handles
            )

            if self.log_goodput:
                with tf.control_dependencies((superchunk_num_records,)):
                    ts = gate.unix_timestamp(name="sort_tail_timestamp")
                log_event = (gate.log_events(
                    item_names=("id", "time", "record_id", "num_records"),
                    directory=self.log_directory,
                    event_name="sort_tail",
                    name="sort_tail_event_logger",
                    components=(slice_id(id_and_count), ts, record_id, superchunk_num_records)
                ),)
            else:
                log_event = ()

            yield (id_and_count, record_id, intermediate_name, superchunk_num_records, superchunk_matrix) + rest_of_input, log_event

    def make_decomp_stage(self, ready_to_decomp):
        """
        :param args:
        :param ready_to_decomp: generator of (id_and_count, [:rest of input], {column_stack} )
        :return: a generator of [ id_and_count, [output_buffer_handles], num_records, first_ordinal, record_id, { rest of input } ]
        """
        ready_to_decomp = sanitize_generator(ready_to_decomp)
        num_columns = len(self.extended_columns)

        # to_agd_reader = just the columns
        # pass_around_agd_reader = (id_and_count, rest, of, input, ...)
        to_agd_reader, pass_around_agd_reader = zip(*(
            (rtd[-1], (rtd[0],)+tuple(rtd[1:-1])) for rtd in ready_to_decomp
        ))

        def gen_timestamps():
            for group in pass_around_agd_reader:
                idc = group[0]
                with tf.control_dependencies((idc,)):
                    ts = gate.unix_timestamp(name="sort_head_timestamp")
                event_log_op = gate.log_events(
                    item_names=("id","time"),
                    components=(slice_id(idc),ts),
                    event_name="sort_head",
                    directory=self.log_directory,
                    name="sort_head_event_logger"
                )
                yield event_log_op

        reader_kwargs = {}
        timestamps = []
        if self.log_goodput:
            timestamps.extend(gen_timestamps())
            assert len(timestamps) == len(ready_to_decomp)
            # control dependencies have to be an iterable
            reader_kwargs["control_ops"] = tuple((a,) for a in timestamps)

        # [output_buffer_handles], num_records, first_ordinal, record_id; in order, for each column group in upstream_tensorz
        repack = [c == base_extension for c in self.extended_columns]
        multi_column_gen = tuple(pipeline.agd_reader_multi_column_pipeline(upstream_tensorz=to_agd_reader, verify=self.deep_verify,
                                                                           repack=repack,
                                                                           name="align_reader", **reader_kwargs))

        for pass_around, generated in zip(pass_around_agd_reader, multi_column_gen):
            yield (pass_around[0],) + tuple(generated) + tuple(pass_around[1:])

    def make_head_gate(self, upstream_gate):
        id_and_count, components = upstream_gate.dequeue_partition(count=self.sort_batch)
        gate_name = "_".join((self.local_dest, "head_gate"))
        head_gate = gate.StreamingGate(limit_upstream=False, limit_downstream=True, # turning both off because there is only one needed, and no credit control is necessary
                                       id_and_count_upstream=id_and_count, sample_tensors=components,
                                       sample_tensors_are_batch=True,
                                       capacity=self.head_gate_capacity,
                                       name=gate_name, shared_name=gate_name)
        enq_ops = (head_gate.enqueue_many(id_and_count=id_and_count, components=components),)
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=head_gate, enqueue_ops=enq_ops))
        return head_gate

    def _make_graph(self, upstream_gate):
        head_gate = self.make_head_gate(upstream_gate=upstream_gate)
        return self.make_graph_impl(local_gate=head_gate)

    @abstractmethod
    def make_graph_impl(self, local_gate):
        raise NotImplementedError

class CephSort(SortCommonStage, Ceph):
    def __init__(self, args):
        super().__init__(args=args)
        self.add_ceph_attrs(args=args)

    @classmethod
    def add_graph_args(cls, parser):
        super().add_graph_args(parser=parser)
        cls.add_ceph_args(parser=parser)

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
            upstream_tensors=(d.components for d in dequeue_ops),
            user_name=self.ceph_user_name,
            cluster_name=self.ceph_cluster_name,
            ceph_conf_path=str(self.ceph_conf_path.absolute()),
            ceph_read_size=self.ceph_read_chunk_size,
            pool_name=self.ceph_pool_name,
            columns=self.extended_columns,
            name="sort_ceph_read",
            **kwargs
        )

        return ((a,)+tuple(b) for a,b in zip((d.id_and_count for d in dequeue_ops), comp_gen))

    def make_write_stage(self, write_ready_inputs):
        """
        :param write_ready_inputs: a generator of [ id_and_count, record_id, intermediate_name, num_recs, superchunk_matrix, {key, namespace} ]
        :return: a gen of [ id_and_count, intermediate_name, namespace, num_recs, [list of written paths, for each column] ]
        """
        if not isinstance(write_ready_inputs, (tuple, list)):
            write_ready_inputs = tuple(write_ready_inputs)

        extensions = self.extended_columns
        name = "sort_ceph_write"
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
                             name=name,
                             num_records=num_recs)

            def gen_written_paths():
                for chunk_file, extension, write_type in zip(chunk_files, extensions, write_types):
                    full_key = tf.string_join([intermediate_name, extension], separator=".", name="full_key_join_{ext}_{t}".format(
                        ext=extension, t=write_type
                    ))
                    result = writer(
                        path=full_key,
                        record_type=write_type,
                        resource_handle=chunk_file,
                        name="intermediate_ceph_writer_{ext}_{t}".format(ext=extension, t=write_type)
                    )
                    out_path = result.output_path
                    if self.log_goodput:
                        timestamp = result.time
                        duration = result.duration
                        bytes = result.bytes
                        log_op = gate.log_events(
                            item_names=("timestamp", "duration", "bytes", "key", "id"),
                            directory=self.log_directory,
                            event_name="sort_ceph_write",
                            name="sort_ceph_write_logger",
                            components=(timestamp, duration, bytes, out_path, slice_id(idc))
                        )
                        with tf.control_dependencies((log_op,)):
                            out_path = tf.identity(out_path)
                    yield out_path

            # have to include the written paths, even though we discard them, so that the ops get triggered
            yield idc, intermediate_name, namespace, num_recs, tuple(gen_written_paths())

    def make_graph_impl(self, local_gate):
        with tf.name_scope("read_stage"):
            read_results = self.make_read_stage(gate=local_gate)

        ready_to_write = self.make_central_pipeline(inputs=read_results, local_head_gate=local_gate)

        with tf.name_scope("write_stage"):
            write_results = self.make_write_stage(write_ready_inputs=ready_to_write)

        queue_name = "completed"
        sink_queue = pipeline.join(upstream_tensors=write_results,
                                   parallel=self.sink_parallel,
                                   multi=True,
                                   capacity=self.final_sink_capacity,
                                   name=queue_name,
                                   shared_name=queue_name)
        return tuple(s[:-1] for s in sink_queue) # :-1 to leave off the file records that aren't needed

