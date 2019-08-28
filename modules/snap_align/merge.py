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
from .fused_align_sort import FusedCommonStage, location_value
from .sort import SortCommonStage
from .snap_align import SnapCommonStage
from .common import *
from common import parse

import tensorflow as tf

persona_ops = tf.contrib.persona.persona_ops()
from tensorflow.contrib.persona import pipeline
import tensorflow.contrib.gate as gate
from abc import abstractmethod

class MergedCommonStage(Stage, ShareArguments):
    local_dest = "merge"

    new_dataset_extension = tf.constant("sorted")

    def __init__(self, args):
        super().__init__()
        expected_args = ("read_parallel", "merge_parallel", "chunk", "compress_parallel", "write_parallel", "sink_parallel", "overwrite",
                         "index_parallel",
                         "log_goodput", "log_directory")
        for expected_arg in expected_args:
            arg_name = "_".join((self.local_dest, expected_arg))
            setattr(self, expected_arg, getattr(args, arg_name))

        if hasattr(args, "log_goodput") and args.log_goodput is True:
            self.log.info("Override enabling log goodput from global param")
            self.log_goodput = args.log_goodput

        defer_to_align_attrs = ("max_secondary", "order_by")
        self.get_arguments_from_other(
            log=self.log,
            this_dest=self.local_dest, other_dest=FusedCommonStage.local_dest,
            args=args, attrs=defer_to_align_attrs
        )
        self.get_arguments_from_other(
            log=self.log,
            this_dest=self.local_dest, other_dest=SortCommonStage.local_dest,
            args=args, attrs=defer_to_align_attrs
        )
        self.get_arguments_from_other(
            log=self.log,
            this_dest=self.local_dest, other_dest=SnapCommonStage.local_dest,
            args=args, attrs=defer_to_align_attrs[:1] # align doesn't have order_by
        )

        results_columns = (results_extension,) + tuple("secondary{}".format(i) for i in range(self.max_secondary))
        self.log.info("merge order by '{ob}'".format(ob=self.order_by))
        if self.order_by == location_value:
            self.columns = results_columns + FusedCommonStage.columns
        else: # by metadata
            self.columns = (metadata_extension, base_extension, qual_extension) + results_columns
        self.results_columns = results_columns

        queue_length_defaults = (
            # these pre-gate capacities are so small because we need to avoid imbalance between machines
            ("head_gate_capacity", "merge_parallel", 0, 1), # in terms of total number of datasets
            ("pre_merge_gate_capacity", "merge_parallel", 0, 1), # in terms of the number of datasets
            ("index_capacity", "index_parallel", 1, 1), # in terms of the number of datasets
            ("post_merge_capacity", "compress_parallel", 1, 0),
            ("pre_write_capacity", "write_parallel", 1, 1),
            ("final_capacity", "sink_parallel", 5, 0), # TODO this queue should basically be empty anyway
        )

        for queue_cap_attr, source_attr, expansion_factor, additive_factor in queue_length_defaults:
            args_attr = "_".join((self.local_dest, queue_cap_attr))
            args_value = getattr(args, args_attr)
            default_value = int(getattr(self, source_attr) * expansion_factor) + int(additive_factor)
            if args_value is None:
                assert default_value > 0, "Computed a non-positive number for the default value of arg '{k}': {v}".format(k=queue_cap_attr, v=default_value)
                args_value = default_value
                setattr(args, args_attr, args_value) # set this again so runtime can write this out correctly
            elif args_value < default_value:
                self.log.warning("Setting the queue capacity '{name}' to {set}. Recommended minimum is {rec}".format(
                    name=queue_cap_attr, set=args_value, rec=default_value
                ))
            setattr(self, queue_cap_attr, args_value)

    @classmethod
    def add_common_graph_args(cls, parser):
        """
        The common ratios needed for an individual machine are the following:

        For every 1 merge parallel, you need
        * 8-9 compress-parallel (9 is more stable, better use this ratio when run the "mixed FAS/M" node)
        * 2 write-parallel
        * 2 read-parallel

        These numbers hit a ceiling as the maximum I/O is hit for a given machine

        :param parser:
        :return:
        """
        prefix = cls.local_dest
        cls.prefix_option(parser=parser, prefix=prefix, argument="read-parallel", default=18, type=numeric_min_checker(1, "must have >= 1 parallel reads"), help="number of parallel read groups to make")
        cls.prefix_option(parser=parser, prefix=prefix, argument="index-parallel", default=10, type=numeric_min_checker(1, "must have >= 1 parallel index building ops"), help="number of parallel groups to build indexes for, if the sort option is by location")
        cls.prefix_option(parser=parser, prefix=prefix, argument="merge-parallel", default=2, type=numeric_min_checker(1, "must have >= 1 parallel merges"), help="number of parallel merges")
        cls.prefix_option(parser=parser, prefix=prefix, argument="chunk", default=100000, type=numeric_min_checker(1, "must have >= 1 parallel merges"), help="size of chunks to output by the merge fused_align_sort")
        cls.prefix_option(parser=parser, prefix=prefix, argument="compress-parallel", default=32, type=numeric_min_checker(1, "must have >= 1 parallel compressors"), help="number of parallel compression stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="write-parallel", default=18, type=numeric_min_checker(1, "must have >= 1 parallel writes"), help="number of writing stages")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sink-parallel", type=numeric_min_checker(1, "must have >0 parallel return stages"), default=4, help="number of parallel stages to return at the end of this pipeline")
        cls.prefix_option(parser=parser, prefix=prefix, argument="overwrite", default=False, action="store_true", help="overwrite the existing dataset instead of making a new _sorted version")

        # deliberately leaving the default unspecified so we can take it from align, if specified as well, to avoid duplicate specification
        cls.prefix_option(parser=parser, prefix=prefix, argument="max-secondary", type=numeric_min_checker(0, "can't have negative secondary results"), help="number of secondary results to expect")
        cls.prefix_option(parser=parser, prefix=prefix, argument="order-by", choices=(location_value, "metadata"), help="sort by this parameter [location | metadata]")

        # all options below here are rather verbose, for length of queues
        cls.prefix_option(parser=parser, prefix=prefix, argument="head-gate-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity for head gate for a given partition")
        cls.prefix_option(parser=parser, prefix=prefix, argument="index-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity for pre-indexing queue")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-merge-gate-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity for pre-merge gate, in terms of number of datasets")
        cls.prefix_option(parser=parser, prefix=prefix, argument="post-merge-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity for post-merge queue")
        cls.prefix_option(parser=parser, prefix=prefix, argument="pre-write-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="length of post-align, pre-write queues")
        cls.prefix_option(parser=parser, prefix=prefix, argument="final-capacity", type=numeric_min_checker(1, "must have >= 1 capacity"), help="capacity for final queue")

        cls.prefix_option(parser=parser, prefix=prefix, argument="log-goodput", default=False, action="store_true", help="log the goodput events")
        cls.prefix_option(parser=parser, prefix=prefix, argument="log-directory", default="/home/whitlock/tf/shell", help="the directory to log all events to, if log_goodput is enabled")

    def make_index_building_stage(self, read_columns):
        queue_name = "index_building_queue"
        read_columns = tuple((a,)+tuple(b) for a,b in read_columns)
        to_convert = pipeline.join(upstream_tensors=read_columns,
                                   parallel=self.index_parallel,
                                   capacity=self.index_capacity,
                                   multi=True,
                                   name=queue_name, shared_name=queue_name)

        pool = persona_ops.results_index_pool(bound=False, size=0)
        for all_components in to_convert:
            idc = all_components[0]
            components = all_components[1:]
            results_column = components[0][0] # first column of chunk matrix
            results_index = persona_ops.results_index_creator(index_pool=pool, column=results_column)
            yield idc, (results_index,) + tuple(components)

    def make_central_pipeline(self, read_columns, head_gate):
        """
        :param read_columns: a generator of (id_and_count, ([ list, of, file, mmap, handles, ... ], {pass around}))
        :return: a generator of (id_and_count, (chunk_matrix, record_id, {pass around})
        """
        read_columns = sanitize_generator(read_columns)

        if self.order_by == location_value:
            read_columns = sanitize_generator(self.make_index_building_stage(read_columns=read_columns))

        # a gen of (id_and_count, components)
        # components = ([ handles, columns ])
        queue_name = "pre_merge_barrier_gate"
        example_idc, example_comp = read_columns[0]
        pre_merge_gate = gate.StreamingGate(
            name=queue_name, shared_name=queue_name,
            id_and_count_upstream=example_idc,
            sample_tensors=example_comp,
            capacity=self.pre_merge_gate_capacity,
            limit_upstream=True, limit_downstream=False
        )
        gate.add_credit_supplier_from_gates(
            upstream_gate=head_gate,
            downstream_gate=pre_merge_gate
        )

        enqueue_ops = tuple(pre_merge_gate.enqueue(
            id_and_count=idc, components=comp
        ) for idc, comp in read_columns)
        gate.add_gate_runner(gate_runner=gate.GateRunner(
            gate=pre_merge_gate,
            enqueue_ops=enqueue_ops
        ))

        to_merge = (
            pre_merge_gate.dequeue_whole_dataset() for _ in range(self.merge_parallel)
        )

        with tf.name_scope("merge_merge_stage"):
            to_compress = tuple(self.make_merge_stage(merge_batches=to_merge))

        with tf.name_scope("merge_compress_stage"):
            to_write_items = tuple(self.make_compress_stage(to_compress=to_compress)) # returns a generator

        control_deps = tuple(a[1] for a in to_write_items)
        to_write_items = tuple(a[0] for a in to_write_items)

        queue_name = "merge_pre_write_queue"
        to_write = pipeline.join(upstream_tensors=to_write_items,
                                 control_dependencies=control_deps,
                                 parallel=self.write_parallel,
                                 capacity=self.pre_write_capacity,
                                 multi=True,
                                 name=queue_name, shared_name=queue_name)
        return to_write

    def make_compress_stage(self, to_compress):
        """
        :param to_compress: a generator of (chunk_handles, num_records, first_ordinal, total_chunks, id_and_count, record_id, {pass around})
        :return: a generator of (id_and_count, chunk_file_matrix, first_ordinal, num_records, record_id, {pass around})
        """
        def compress_pipeline(handles):
            with tf.name_scope("merge_compress_results"):
                buffer_pool = persona_ops.buffer_pool(bound=False, size=10)

                compressors = tuple(partial(persona_ops.buffer_pair_compressor, buffer_pool=buffer_pool, pack=False, name="buffer_pair_compressor_{}".format(cname)) for cname in self.columns)
                for buffer_pairs in handles:
                    bps_unstacked = tf.unstack(buffer_pairs)
                    compressed_buffers = tuple(compressor(buffer_pair=a) for compressor, a in zip(compressors, bps_unstacked))

                    def gen_buffers(bufs):
                        for cb in bufs:
                            compressed_buffer = cb.compressed_buffer
                            if self.log_goodput:
                                timestamp = cb.time
                                duration = cb.duration
                                original_size = cb.original_size
                                compressed_size = cb.compressed_size
                                log_op = gate.log_events(
                                    item_names=("timestamp", "duration", "original_bytes", "compressed_bytes"),
                                    directory=self.log_directory,
                                    event_name="merge_compression",
                                    name="merge_compression_logger",
                                    components=(timestamp, duration, original_size, compressed_size)
                                )
                                with tf.control_dependencies((log_op,)):
                                    compressed_buffer = tf.identity(compressed_buffer)
                            yield compressed_buffer

                    yield tf.stack(tuple(gen_buffers(bufs=compressed_buffers)))

        to_compress = sanitize_generator(to_compress)
        for chunk_file_matrix, (num_records, first_ordinal, total_num_chunks, id_and_count, record_id), pass_around in \
            zip(
                compress_pipeline(handles=(a[0] for a in to_compress)),
                (a[1:6] for a in to_compress),
                (a[6:] for a in to_compress)
            ):
            ids_only = tf.unstack(id_and_count, axis=1, name="id_only_extractor")[0]
            new_count = tf.fill(ids_only.shape, total_num_chunks, name="new_counts_fill")
            new_id_and_count = tf.stack((ids_only, new_count), axis=1, name="new_id_and_count_constructor")
            control_deps = []
            if self.log_goodput:
                with tf.control_dependencies((new_id_and_count,)):
                    ts = gate.unix_timestamp()
                control_deps.append(
                    gate.log_events(
                        item_names=("id", "time", "record_id", "num_records"),
                        event_name="merge_tail",
                        directory=self.log_directory,
                        components=(ids_only,ts,record_id,num_records)
                    )
                )

            yield (new_id_and_count, chunk_file_matrix, first_ordinal, num_records, record_id) + tuple(pass_around), control_deps

    def make_merge_stage(self, merge_batches):
        """
        :param merge_batches: a generator of (id_and_count, ([maybe results index], chunk_matrix, record_id, {pass around})), as entire batches
        :return: a generator of (chunk_handles, num_records, first_ordinal, total_chunks, id_and_count, record_id, {pass around})
        """
        merge_batches = sanitize_generator(merge_batches)

        assert len(merge_batches) > 0
        slice_index = 2
        if self.order_by == location_value:
            slice_index += 1
        comp_extra_example = merge_batches[0][1][slice_index:]
        component_dtypes = (tf.string, tf.int32, tf.int64,) + (tf.int32,)*2 + (tf.string,) + tuple(a.dtype for a in comp_extra_example)
        shapes = (tf.TensorShape([len(self.columns), 2]),) + (tf.TensorShape([]),)*3 + (merge_batches[0][0].shape,) + (tf.TensorShape([]),) +\
                 tuple(a[0].shape for a in comp_extra_example) # a[0].shape because we only take the first slice of the batches
        assert len(shapes) == len(component_dtypes)

        queue_name = "post_merge_queue"
        queue = tf.FIFOQueue(
            dtypes=component_dtypes,
            shapes=shapes,
            capacity=self.post_merge_capacity, # post_merge_capacity
            name=queue_name, shared_name=queue_name
        )
        tf.summary.scalar(
           "queue/{n}/fraction_of_{cap}_full".format(n=queue_name, cap=self.post_merge_capacity),
            tf.cast(queue.size(), tf.float32) * (1. / self.post_merge_capacity)
        )

        # bpp = persona_ops.buffer_pair_pool(size=0, bound=False, name="merge_buffer_list_pool")
        bpp = persona_ops.primed_buffer_pair_pool(
            bound=False,
            name="merge_primed_buffer_list_pool",
            num_records=self.chunk,
            record_size=101, # longest are the quals
            size=len(self.columns) * (self.merge_parallel + self.post_merge_capacity + self.compress_parallel)
        )

        self.log.info("order by is still '{ob}'".format(ob=self.order_by))
        if self.order_by == location_value:
            merge_op = persona_ops.agd_merge
        else:
            raise Exception("metadata merging not supported")
            merge_op = persona_ops.agd_merge_metadata

        merge = partial(merge_op, chunk_size=self.chunk,
                        buffer_pair_pool=bpp,
                        output_buffer_queue_handle=queue.queue_ref,
                        name="agd_merge")

        # remember, all of components is a mega batch!
        def make_merge_ops():
            for id_and_count, components in merge_batches:
                merge_kwargs = {}
                if self.order_by == location_value:
                    merge_kwargs["results_indexes"] = components[0]
                    components = components[1:]
                column_handless, record_ids = components[:2]
                pass_around = tuple(a[0] for a in components[2:]) # slice so we only get the frist component
                record_id = record_ids[0]
                control_deps = []
                if self.log_goodput:
                    with tf.control_dependencies((id_and_count,)): # put it after this in case the merge takes a while
                        ts = gate.unix_timestamp()
                    control_deps.append(
                        gate.log_events(
                            item_names=("id", "time"),
                            directory=self.log_directory,
                            event_name="merge_head",
                            components=(slice_id(id_and_count), ts)
                        )
                    )
                with tf.control_dependencies(control_deps):
                    yield merge(
                        chunk_group_handles=column_handless,
                        other_components=(id_and_count,record_id)+tuple(pass_around),
                        **merge_kwargs
                    )
        merge_ops = tuple(make_merge_ops())
        tf.train.add_queue_runner(qr=tf.train.QueueRunner(
            queue=queue, enqueue_ops=merge_ops
        ))

        return (queue.dequeue(name="merge_queue_dequeue") for _ in range(self.compress_parallel))

    def process_stage(self, data_columns):
        """
        :param data_columns: a generator of (list, of, column, handles, to, be, processed)
        :return: a generator of ([ handles ]), where [handles] is a matrix
        """
        # [output_buffer_handles], record_id; in order, for each column group in upstream_tensorz
        pool = persona_ops.raw_file_system_column_pool(bound=False, size=0)
        convert = partial(persona_ops.raw_file_converter, column_pool=pool)
        for handles in data_columns:
            values = tuple(convert(data=file_handle) for file_handle in handles)
            handles, record_ids = zip(*values)
            assert len(record_ids) > 0
            yield tf.stack(handles, name="stack_raw_filesystem_columns"), record_ids[0]

    def make_head_gate(self, upstream_gate):
        id_and_count, components = upstream_gate.dequeue_whole_partition()
        gate_name = "merge_head_gate"
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
        local_gate = self.make_head_gate(upstream_gate=upstream_gate)
        return self.make_graph_impl(local_gate=local_gate)

    @abstractmethod
    def make_graph_impl(self, local_gate):
        raise NotImplementedError

class LocalMergeStage(MergedCommonStage):

    @classmethod
    def add_graph_args(cls, parser):
        cls.add_common_graph_args(parser=parser)

    def make_read_stage(self, local_gate):
        """

        :param local_gate: components in local_gate: just the basename of intermediate files
        :return: a generator of type (id_and_count, [ handles, to, mmaped, columns ], filenames)
        """
        def gen_filenames():
            for i in range(self.read_parallel):
                idc, comp = local_gate.dequeue()
                assert len(comp) == 1
                filename = comp[0]
                yield idc, filename
        ids_and_counts, filenames = zip(*gen_filenames())
        assert len(filenames) > 0
        read_groups =(tuple(a) for a in pipeline.local_read_pipeline(
            delete_after_use=True, upstream_tensors=filenames, columns=self.columns, name="local_read_merge_pipeline"
        ))

        pool = persona_ops.raw_file_system_column_pool(bound=False, size=0)
        convert = partial(persona_ops.raw_file_converter, column_pool=pool)

        def gen_conversion():
            for read_group in read_groups:
                values = tuple(convert(data=file_handle) for file_handle in read_group)
                handles, record_ids = zip(*values)
                assert len(record_ids) > 0
                yield tf.stack(handles, name="stack_raw_filesystem_columns"), record_ids[0]

        for idc, (handles, record_id), filename in zip(ids_and_counts, gen_conversion(), filenames):
            yield idc, (handles, record_id, filename)

    def make_write_stage(self, ready_to_write_items):
        """
        :param ready_to_write_items: a generator of (id_and_count, chunk_file_matrix, first_ordinal, num_records, record_id, filename)
        :return: a generator of (id_and_count, record_id, first_ordinal, num_records, file_basename) + (list, of, full, file, paths)
        """
        write_items = tuple(ready_to_write_items) # need to iterate multiple times
        write_items = tuple(
            (id_and_count, buffer_handles, first_ordinal, num_records, record_id, file_directory)
            for (id_and_count, buffer_handles, first_ordinal, num_records, record_id), file_directory in
            zip(
                (a[:5] for a in write_items),
                (dirname(filename=b[5]) for b in write_items)
            )
        )
        chunk_basenames = (
            (tf.string_join((file_directory, record_id), separator=path_separator_str, name="chunk_base_join"), tf.as_string(first_ordinal))
            for id_and_count, buffer_handles, first_ordinal, num_records, record_id, file_directory in write_items
        )
        if self.overwrite:
            chunk_basenames = tuple(
                tf.string_join((chunk_base, ordinal_as_string), separator="_", name="final_chunk_join")
                for chunk_base, ordinal_as_string in chunk_basenames
            )
        else:
            chunk_basenames = tuple(
                tf.string_join((chunk_base, self.new_dataset_extension, ordinal_as_string), separator="_", name="final_chunk_join")
                for chunk_base, ordinal_as_string in chunk_basenames
            )
        to_writer_gen = (
            (buffer_handles, record_id, first_ordinal, num_records, file_basename)
            for (id_and_count, buffer_handles, first_ordinal, num_records, record_id), file_basename in zip(
                (a[:-1] for a in write_items),
                chunk_basenames
            )
        )
        around_writer_gen = (
            (id_and_count, record_id, first_ordinal, num_records, file_basename)
            for (id_and_count, buffer_handles, first_ordinal, num_records, record_id), file_basename in zip(
                (a[:-1] for a in write_items),
                chunk_basenames
            )
        )

        written_records = (
            tuple(a) for a in pipeline.local_write_pipeline(upstream_tensors=to_writer_gen,
                                                            compressed=True, # compressed controls the buffer output
                                                            record_types=(get_dicts_for_extension(self.columns,
                                                                                                  text_base=False)))
        )
        results = tuple(a+b for a,b in zip(
            around_writer_gen, written_records
        ))
        assert len(results) == len(write_items)
        return results

    def make_graph_impl(self, local_gate):
        # :return: a generator of (id_and_count, record_id, first_ordinal, num_records, file_basename) + (list, of, full, file, paths
        with tf.name_scope("merge_read"):
            ready_to_merge_items = self.make_read_stage(local_gate=local_gate)

        with tf.name_scope("merge"):
            ready_to_write_items = self.make_central_pipeline(read_columns=ready_to_merge_items,
                                                              head_gate=local_gate)

        with tf.name_scope("merge_write"):
            completed_items = self.make_write_stage(ready_to_write_items=ready_to_write_items)

        final_name = "merge_completed_items_queue"
        return pipeline.join(upstream_tensors=completed_items,
                             parallel=self.sink_parallel,
                             capacity=self.final_capacity,
                             multi=True,
                             name=final_name, shared_name=final_name)

class CephMergeStage(MergedCommonStage, Ceph):
    def __init__(self, args):
        super().__init__(args=args)
        self.add_ceph_attrs(args=args)

        lazy_attrs = ("ceph_num_lazy_segments", "ceph_records_per_segment", "lazy_ceph")
        for la in lazy_attrs:
            setattr(self, la, getattr(args, la))

    @classmethod
    def add_graph_args(cls, parser):
        cls.add_common_graph_args(parser=parser)
        cls.add_ceph_args(parser=parser)
        # Note: this option is inverted. lazy ceph reads are the default
        if parser.get_default("lazy_ceph") is None:
            parser.add_argument("--eager-ceph", dest="lazy_ceph", default=True, action="store_false", help="use the eager read version of the ceph pipeline (not lazy)")
            parser.add_argument("--ceph-lazy-records-per-segment", dest="ceph_records_per_segment", default=250000, type=parse.numeric_min_checker(
                minimum=1000, message="minimum of 1000 for records per segment"
            ), help="number of records for each segment in the ceph lazy reader")
            parser.add_argument("--ceph-lazy-segments", dest="ceph_num_lazy_segments", default=2, type=parse.numeric_min_checker(
                minimum=1, message="must have at least one lazy segment"
            ), help="number of lazy segments for each asynchronous ceph lazy column")

    def _gen_common_read(self, local_gate):
        for i in range(self.read_parallel):
            idc, comp = local_gate.dequeue()
            assert len(comp) == 2
            yield idc, comp

    def make_eager_read_stage(self, local_gate):
        """
        :param local_gate: components in local_gate: [ id_and_count, intermediate_name, namespace ]
        :return: a generator of type (id_and_count, [ handles, to, mmaped, columns ], namespace)
        """
        raise Exception("We're not supporting eager ceph! Have to throw here to ensure that!")
        ids_and_counts, keys_and_namespaces = zip(*self._gen_common_read(local_gate=local_gate))
        assert len(keys_and_namespaces) > 0
        pool = persona_ops.raw_file_system_column_pool(bound=False, size=0)
        convert = partial(persona_ops.raw_file_converter, column_pool=pool)
        kwargs = {}
        if self.log_goodput:
            kwargs["log_directory"] = self.log_directory
            kwargs["metadata"] = tuple(slice_id(idc) for idc in ids_and_counts)
        def gen_read_groups():
            for k,nmspc,chunk_buffers in pipeline.ceph_read_pipeline(
                delete_after_read=True,
                upstream_tensors=keys_and_namespaces,
                user_name=self.ceph_user_name,
                cluster_name=self.ceph_cluster_name,
                ceph_conf_path=str(self.ceph_conf_path),
                pool_name=self.ceph_pool_name,
                ceph_read_size=self.ceph_read_chunk_size,
                columns=self.columns,
                name="merge_ceph_read",
                **kwargs
            ):
                values = tuple(convert(data=buffer_handle) for buffer_handle in chunk_buffers)
                handles, record_ids = zip(*values)
                assert len(record_ids) > 0
                yield tf.stack(handles, name="stack_ceph_handles"), record_ids[0]

        for idc, (handles, record_id), namespace in zip(ids_and_counts, gen_read_groups(), (a[1] for a in keys_and_namespaces)):
            yield idc, (handles, record_id, namespace) # namespace is part of the passed-around stuff

    def make_lazy_read_stage(self, local_gate):
        ids_and_counts, keys_and_namespaces = zip(*self._gen_common_read(local_gate=local_gate))
        assert len(keys_and_namespaces) > 0

        for id_and_count, (key, namespace, chunk_buffers, record_id) in zip(
            ids_and_counts, pipeline.ceph_combo_read_pipeline(
                    upstream_tensors=keys_and_namespaces,
                    user_name=self.ceph_user_name,
                    cluster_name=self.ceph_cluster_name,
                    ceph_conf_path=self.ceph_conf_path,
                    pool_name=self.ceph_pool_name,
                    columns=self.columns,
                    eager_column_types=set(self.results_columns),
                    records_per_segment=self.ceph_records_per_segment,
                    segments_to_buffer=self.ceph_num_lazy_segments,
                    delete_after_read=True,
                    name="merge_lazy_ceph_read_pipeline"
                )
        ):
            yield id_and_count, (chunk_buffers, record_id, namespace)

    def make_write_stage(self, ready_to_write_items):
        """
        :param ready_to_write_items: a generator of (id_and_count, chunk_file_matrix, first_ordinal, num_records, record_id, namespace)
        :return: a generator of (id_and_count, record_id, first_ordinal, num_records, key_basename, namespace) + (list, of, full, file, keys)
        """
        write_items = sanitize_generator(ready_to_write_items) # need to iterate multiple times
        chunk_basenames = (
            (record_id, tf.as_string(first_ordinal))
            for id_and_count, buffer_handles, first_ordinal, num_records, record_id, namespace in write_items
        )
        if self.overwrite:
            chunk_basenames = tuple(
                tf.string_join((chunk_base, ordinal_as_string), separator="_", name="final_chunk_join")
                for chunk_base, ordinal_as_string in chunk_basenames
            )
        else:
            chunk_basenames = tuple(
                tf.string_join((chunk_base, self.new_dataset_extension, ordinal_as_string), separator="_", name="final_chunk_join")
                for chunk_base, ordinal_as_string in chunk_basenames
            )

        to_writer_gen = (
            (key, namespace, num_records, first_ordinal, record_id, buffer_handles)
            for key, (id_and_count, buffer_handles, first_ordinal, num_records, record_id, namespace) in zip(
                chunk_basenames, write_items
            )
        )

        around_writer_gen = (
            (id_and_count, record_id, first_ordinal, num_records, key, namespace)
            for (id_and_count, buffer_handles, first_ordinal, num_records, record_id, namespace), key in zip(
                write_items,
                chunk_basenames
            )
        )

        kwargs = {}
        if self.log_goodput:
            kwargs["log_directory"] = self.log_directory
            kwargs["metadata"] = tuple(slice_id(a[0]) for a in write_items)

        written_records = (
            tuple(a) for a in pipeline.ceph_write_pipeline(upstream_tensors=to_writer_gen,
                                                           compressed=True,
                                                           user_name=self.ceph_user_name,
                                                           cluster_name=self.ceph_cluster_name,
                                                           pool_name=self.ceph_pool_name,
                                                           name="merge_ceph_write",
                                                           ceph_conf_path=str(self.ceph_conf_path),
                                                           record_types=(get_dicts_for_extension(self.columns,
                                                                                                 text_base=False)),
                                                           **kwargs)
        )
        results = tuple(a+b for a,b in zip(
            around_writer_gen, written_records
        ))
        assert len(results) == len(write_items)
        return results

    def make_graph_impl(self, local_gate):
        # :return: a generator of (id_and_count, record_id, first_ordinal, num_records, key_basename, namespace) + (list, of, full, file, keys)
        with tf.name_scope("merge_read"):
            if self.lazy_ceph:
                ready_to_merge_items = tuple(self.make_lazy_read_stage(local_gate=local_gate))
            else:
                ready_to_merge_items = self.make_eager_read_stage(local_gate=local_gate)

        with tf.name_scope("merge"):
            ready_to_write_items = self.make_central_pipeline(read_columns=ready_to_merge_items,
                                                              head_gate=local_gate)

        with tf.name_scope("merge_write"):
            completed_items = self.make_write_stage(ready_to_write_items=ready_to_write_items)

        final_name = "merge_completed_items_queue"
        return gate.pipeline.streaming_gate(
            id_and_count_components=((a[0], a[1:]) for a in completed_items),
            parallel=self.sink_parallel,
            name=final_name, shared_name=final_name
        )
        # return pipeline.join(upstream_tensors=completed_items,
        #                      parallel=self.sink_parallel,
        #                      capacity=self.final_capacity,
        #                      multi=True,
        #                      name=final_name, shared_name=final_name)

# small versions

class SmallMergedCommonStage(MergedCommonStage):
    local_dest = "{}small".format(MergedCommonStage.local_dest)

    default_parallelism_args = (
        ("read_parallel", 2),
        ("merge_parallel", 2),
        ("compress_parallel", 14),
        ("write_parallel", 4),
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

    from_fas = ("max_secondary", "order_by")
    from_big_merge = ("overwrite",)
    def __init__(self, args):
        super().__init__(args=args)
        self.get_arguments_from_other(attrs=self.from_big_merge+self.from_fas,
                                      this_dest=self.local_dest,
                                      other_dest=MergedCommonStage.local_dest,
                                      args=args, log=self.log)
        # Need to "re-get" these arguments from FusedAlignSort, if they exist
        self.get_arguments_from_other(attrs=self.from_fas,
                                      this_dest=self.local_dest,
                                      other_dest=FusedCommonStage.local_dest,
                                      args=args, log=self.log)
        self.get_arguments_from_other(attrs=self.from_fas,
                                      this_dest=self.local_dest,
                                      other_dest=SortCommonStage.local_dest,
                                      args=args, log=self.log)

class SmallCephMergeStage(SmallMergedCommonStage, CephMergeStage):
    def __init__(self, args):
        super().__init__(args=args)

class SmallLocalMergeStage(SmallMergedCommonStage, LocalMergeStage):
    def __init__(self, args):
        super().__init__(args=args)

# null stages
# mainly to clean up intermediate files

class NullMergeStage(Stage, Ceph, ShareArguments):
    local_dest = "nullceph"

    def __init__(self, args):
        super().__init__()
        expected_args = ("global_batch", "delete_parallel", "sink_parallel")
        for expected_arg in expected_args:
            arg_name = "_".join((self.local_dest, expected_arg))
            setattr(self, expected_arg, getattr(args, arg_name))
        self.add_ceph_attrs(args=args)
        self.get_arguments_from_other(
            attrs=("max_secondary",), this_dest=self.local_dest,
            other_dest=FusedCommonStage.local_dest, args=args, log=self.log
        )
        self.get_arguments_from_other(
            attrs=("max_secondary",), this_dest=self.local_dest,
            other_dest=SortCommonStage.local_dest, args=args, log=self.log
        )
        self.columns  = (results_extension,) + tuple("secondary{}".format(i) for i in range(self.max_secondary)) + FusedCommonStage.columns

    @classmethod
    def add_graph_args(cls, parser):
        prefix = cls.local_dest
        cls.prefix_option(parser=parser, prefix=prefix, default=10, argument="global-batch", type=numeric_min_checker(1, "must have >=1 batch from global gate"), help="batch size for dequeuing from the upstream central gate. Doesn't affect correctness")
        cls.prefix_option(parser=parser, prefix=prefix, default=8, argument="delete-parallel", type=numeric_min_checker(1, "must have >=1 parallel deletion node"), help="number of parallel deletion nodes")
        cls.prefix_option(parser=parser, prefix=prefix, argument="sink-parallel", type=numeric_min_checker(1, "must have >0 parallel return stages"), default=2, help="number of parallel stages to return at the end of this pipeline")
        cls.add_ceph_args(parser=parser)

    def _make_graph(self, upstream_gate):
        def gen_delete_ops():
            remove_op = partial(persona_ops.ceph_remove,
                                cluster_name=self.ceph_cluster_name,
                                user_name=self.ceph_user_name,
                                pool_name=self.ceph_pool_name,
                                columns=self.columns,
                                ceph_conf_path=str(self.ceph_conf_path))
            for idx in range(self.delete_parallel):
                id_and_count, components = upstream_gate.dequeue_many(count=self.global_batch)
                keys, namespaces = components
                num_items_deleted = remove_op(keys=keys, namespaces=namespaces)
                yield id_and_count, num_items_deleted

        items = tuple(gen_delete_ops())
        queue_name = "nullceph_final"
        return pipeline.join(upstream_tensors=items,
                             parallel=self.sink_parallel,
                             capacity=self.delete_parallel+1,
                             multi=True,
                             name=queue_name, shared_name=queue_name)

