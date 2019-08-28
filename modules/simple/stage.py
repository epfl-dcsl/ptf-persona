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
from common.parse import numeric_min_checker
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)
import tensorflow as tf
from tensorflow.contrib.persona import pipeline
import tensorflow.contrib.gate as gate

class Incrementer(Stage):

    def __init__(self, args):
        super().__init__()
        self.increment = args.increment
        self.queue_chain_length = args.queue_chain
        self.parallel = args.parallel_chains

    @classmethod
    def add_graph_args(cls, parser):
        parser.add_argument("--increment", type=numeric_min_checker(minimum=0, message="must increment by a positive amount"), default=1, help="amount to increment by")
        parser.add_argument("--queue-chain", type=numeric_min_checker(minimum=0, message="must have non-negative queue chain length"), default=1, help="length of local queue length (with queue runners)")
        parser.add_argument("--parallel-chains", type=numeric_min_checker(minimum=1, message="must have >=1 chains"), default=1, help="number of chains to run in parallel")

    @staticmethod
    def make_local_gate(upstream_gate):
        idc, comp = upstream_gate.dequeue_partition(count=1)
        head_gate = gate.StreamingGate(limit_upstream=False, limit_downstream=False,
                                       id_and_count_upstream=idc, sample_tensors=comp,
                                       sample_tensors_are_batch=True)
        enq_ops = (head_gate.enqueue_many(id_and_count=idc, components=comp),)
        gate.add_gate_runner(gate_runner=gate.GateRunner(gate=head_gate, enqueue_ops=enq_ops))
        return head_gate

    def even_simpler(self, upstream_gate):
        local_gate = self.make_local_gate(upstream_gate=upstream_gate)
        idc, comp = local_gate.dequeue()
        name = "chain_{cid}_join".format(cid=0)
        idcs_and_comps = pipeline.join(upstream_tensors=(idc, comp),
                                       capacity=1, parallel=1, name=name, shared_name=name)
        return idcs_and_comps[0]


    def _make_graph(self, upstream_gate):
        return self.even_simpler(upstream_gate=upstream_gate)
        increment_constant = tf.constant(self.increment)
        def make_chain(idc, comp, chain_id):
            for idx in range(self.queue_chain_length):
                comp = comp + increment_constant
                idc, comp = pipeline.join(upstream_tensors=(idc, comp),
                                          capacity=2, parallel=1, name="chain_{cid}_join_{idx}".format(idx=idx, cid=chain_id))[0]
            return idc, comp

        local_gate = self.make_local_gate(upstream_gate=upstream_gate)
        idc, comp = local_gate.dequeue() # has to be split up or join() complains :/
        idcs_and_comps = pipeline.join(upstream_tensors=(idc,comp),
                                       parallel=self.parallel,
                                       capacity=self.parallel*2,
                                       name="local_head_gate")

        chains = (make_chain(idc=idc, comp=comp, chain_id=idx) for idx, (idc, comp) in enumerate(idcs_and_comps))

        final = pipeline.join(upstream_tensors=chains,
                              parallel=1, multi=True,
                              capacity=self.parallel*2,
                              name="local_tail_gate")
        return final[0] # just return idc, comp