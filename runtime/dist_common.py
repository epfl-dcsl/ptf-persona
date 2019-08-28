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
# things shared between cluster and client
# such as for the Pyro4 naming
import socket
from common import parse

system_name = "tfns"
pyro_worker_prefix = "{}.slave".format(system_name)
pyro_master_name = "{}.master".format(system_name)

results_key = "results"

def make_tf_device_name(job_name, task_index):
    return "/job:{name}/task:{idx}".format(
        name=job_name, idx=task_index
    )

def add_common_record_args(parser):
    parser.add_argument("--record-stats", default=False, action="store_true", help="store statistics for this process into the output directory")
    parser.add_argument("--output-directory", default="output", type=parse.path_exists_checker(make_if_empty=True, rm_if_exists=True),
                        help="directory to record output of all sorts into. will be made if doesn't exist")

def get_external_ip_addr():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("iccluster001", 22))
        return s.getsockname()[0]
    finally:
        s.close()

def get_tf_dist_info(current_reservations=None, start_port=30000):
    external_ip = get_external_ip_addr()
    def check_port(p):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((external_ip, p))
            return True
        except socket.error as e:
            if e.errno == 98:
                return False
            else:
                raise e
        finally:
            s.close()

    if current_reservations is None:
        current_reservations = dict()
    assert isinstance(current_reservations, dict)

    hostname = socket.gethostname()
    used_ports = set(current_reservations.get(hostname)) if hostname in current_reservations else set()

    while True:
        if start_port not in used_ports and check_port(start_port):
            return {
                "host": socket.gethostname(),
                "port": start_port
            }
        start_port+=1
