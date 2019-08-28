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
import abc
import logging
class Stage(abc.ABC):
    def __init__(self, log_level=logging.DEBUG):
        log = logging.getLogger(self.__class__.__name__)
        log.setLevel(level=log_level)
        self.log = log
        self._graph_outputs = ()

    @property
    def run_first(self):
        return ()

    """
    Constructs a fused_align_sort for TFNS. This graph is to remain entirely within one process,
    though multiple stages may be chained in the same process.
    """
    @classmethod
    @abc.abstractmethod
    def add_graph_args(cls, parser):
        """
        Add arguments for how this fused_align_sort should be constructed in terms of its graph
        :param parser: an argparse.ArgumentParser
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _make_graph(self, upstream_gate):
        """
        Make the graph, based on the parameters in args.
        :param upstream_gate: the GLOBAL upstream gate to this fused_align_sort. Needs to be global because a give Stage may want EVERYTHING or just one thing, etc
        :return: a list of operations to enqueue into the downstream gate. They should all be the same type
        """
        raise NotImplementedError

    def make_graph(self, upstream_gate):
        if len(self._graph_outputs) == 0:
            # make sure this is only created once!
            self._graph_outputs = self._make_graph(upstream_gate=upstream_gate)
        return self._graph_outputs

    @classmethod
    def prefix_option(cls, parser, prefix, argument, *args, **kwargs):
        arg_name = "--{prefix}-{arg}".format(prefix=prefix, arg=argument)
        dest_name = argument.replace("-","_")
        dest_name = "_".join((prefix, dest_name))
        parser.add_argument(arg_name, dest=dest_name, *args, **kwargs)
