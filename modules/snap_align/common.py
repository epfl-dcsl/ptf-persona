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
import pathlib
from common.parse import non_empty_string_checker, path_exists_checker, numeric_min_checker

from string import digits

path_separator_str = "/"
path_separator = tf.constant(path_separator_str)

intermediate_extension = tf.constant("intermediate")
structed_extension_type = "structured"
text_extension_type = "text"
base_extension_type = "base_compact"
results_extension = "results"
secondary_results_extension = "secondary"
metadata_extension = "metadata"
base_extension = "base"
qual_extension = "qual"

type_mapping = {
    results_extension: structed_extension_type,
    secondary_results_extension: structed_extension_type,
    base_extension: base_extension_type,
    qual_extension: text_extension_type,
    metadata_extension: text_extension_type
}

def dirname(filename):
    split_filename = tf.string_split((filename,), delimiter=path_separator_str)
    chunkfile_dir = split_filename.values[:-1]
    joined = tf.reduce_join(chunkfile_dir, separator=path_separator_str, name="path_join_after_slice")
    correct_path = tf.string_join((path_separator, joined), name="correct_path")
    return correct_path

def get_type_for_extension(column_extension, text_base=False):
    assert isinstance(column_extension, str)
    processed = column_extension.translate({ord(k): None for k in digits})
    if processed not in type_mapping:
        raise ValueError("Extension '{e}' not found in mapping".format(e=column_extension))
    if column_extension is base_extension and text_base:
        return text_extension_type
    else:
        return type_mapping[processed]

def get_dict_for_extension(column_extension, *args, **kwargs):
    return {
        "extension": column_extension,
        "type": get_type_for_extension(column_extension=column_extension, *args, **kwargs)
    }

def get_dicts_for_extension(column_extensions, **kwargs):
    for ce in column_extensions:
        yield get_dict_for_extension(column_extension=ce, **kwargs)

class ShareArguments:

    def get_arguments_from_other(self, attrs, this_dest, other_dest, args, log, override=True):
        for a in attrs:
            other_arg_name = "_".join((other_dest, a))
            # this_arg_name = "_".join((this_dest, a))
            if hasattr(args, other_arg_name):
                other_val = getattr(args, other_arg_name)
                this_val = getattr(self, a, None)
                if this_val is None:
                    log.warning("self has no attribute '{a}'. Taking directly from '{od}'".format(a=a, od=other_dest))
                    setattr(self, a, other_val)
                    setattr(args, this_dest, other_val)
                elif this_val != other_val:
                    if override:
                        log.info("overriding attribute '{a}' directly from '{od}'. Old: '{old}'. New: '{new_value}'".format(a=a, od=other_dest,
                                                                                                                            old=this_val, new_value=other_val))
                        setattr(self, a, other_val)
                        setattr(args, this_dest, other_val)
                    else:
                        log.warning("NOT OVERRIDING unequal attributes for '{a}'. This ('{t}') value: {tv}. Other ('{od}') value: {other}".format(
                            a=a, t=this_dest, od=other_dest,
                            tv=this_val, other=other_val
                        ))
                else:
                    log.debug("attributes equivalent for '{a}' between '{this}' and '{od}'".format(a=a, this=this_dest, od=other_dest))
            else:
                log.info("Can't override attribute '{a}' from other dest '{od}'".format(a=a, od=other_dest))

class Ceph:
    ceph_attributes = tuple(
        "_".join(("ceph", a)) for a in (
            "cluster_name",
            "user_name",
            "pool_name",
            "conf_path",
            "read_chunk_size"
        )
    )

    full_ceph_attributes = (
        {
            "attribute": "ceph_cluster_name",
            "type": non_empty_string_checker,
            "default": "ceph",
            "help": "name for the ceph cluster",
        },
        {
            "attribute": "ceph_user_name",
            "type": non_empty_string_checker,
            "default": "client.dcsl1024",
            "help": "ceph username",
        },
        {
            "attribute": "ceph_pool_name",
            "type": non_empty_string_checker,
            "default": "dcsl1024",
            "help": "ceph pool name",
        },
        {
            "attribute": "ceph_conf_path",
            "type": path_exists_checker(check_dir=False),
            "default": "/etc/ceph/ceph.conf",
            "help": "ceph_configuration_path",
        },
        {
            "attribute": "ceph_read_chunk_size",
            "type": numeric_min_checker(128, "must have a reasonably large minimum read size from Ceph"),
            "default": (2**26),
            "help": "minimum size to read from ceph storage, in bytes",
        },
    )

    @classmethod
    def add_ceph_args(cls, parser):
        for attr_dict in cls.full_ceph_attributes:
            attr_name = attr_dict["attribute"]
            if parser.get_default(attr_name) is None:
                arg_name = "--{}".format(attr_name.replace("_","-"))
                parser.add_argument(arg_name, dest=attr_name, type=attr_dict["type"], default=attr_dict["default"], help=attr_dict["help"])

    def add_ceph_attrs(self, args):
        for ceph_attr in self.ceph_attributes:
            setattr(self, ceph_attr, getattr(args, ceph_attr))

def slice_id(id_and_count):
    return tf.squeeze(id_and_count[:, :1], axis=1)

def sanitize_generator(x):
    try:
        x[0]
    except TypeError:
        x = tuple(x)
    return x
