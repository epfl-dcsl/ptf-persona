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
from . import runtime
from . import dist_common
import Pyro4

class ClientRuntime(runtime.Runtime):

    @staticmethod
    def name():
        return "client"

    @staticmethod
    def help_message():
        return "run an application on an existing cluster"

    @classmethod
    def add_arguments(cls, parser):
        cls.add_record_args(parser=parser)
        parser.add_argument("--master", default=dist_common.pyro_master_name, help="Pyro4 name to connect to the master for this service")
        parser.add_argument("--pyro-ns-port", type=int, help="override default Pyro4 nameserver port")
        parser.add_argument("--pyro-ns-host", help="override default Pyro4 nameserver port")

    def get_master(self, name, pyro_ns_host, pyro_ns_port):
        with Pyro4.locateNS(host=pyro_ns_host, port=pyro_ns_port) as ns:
            items = ns.list(prefix=name)
            num_items = len(items)
            if num_items == 0:
                raise Exception("Pyro4 found no master registered at the name '{name}'".format(name=name))
            if num_items > 1:
                self.log.warning("Pyro4 found {n} master objects. This is probably an error! Choosing the first one".format(n=num_items))
            return Pyro4.Proxy(tuple(items.values())[0])

    def _run_application(self, ApplicationClass, args):
        if args.record_args:
            self.write_out_args(args=args)
        app_master = self.get_master(name=args.master,
                                     pyro_ns_host=args.pyro_ns_host,
                                     pyro_ns_port=args.pyro_ns_port)
        ingress_args = ApplicationClass.process_ingress_args(args=args)
        app_name = ApplicationClass.name()
        master_app_name = app_master.app_name
        if app_name != master_app_name:
            raise Exception("Master app name '{mname}' is not equal to this app's name '{cname}'".format(
                mname=master_app_name, cname=app_name
            ))
        # just mocking this out. will likely need a bit of rewriting
        results = app_master.run_client_request(ingress_args=ingress_args)
        ApplicationClass.process_egress_results(results=results.pop(dist_common.results_key), args=args)
        return results # kind of a hack to work with benchmark. normal client should discard this

    @staticmethod
    def _populate_app_args(parser, app):
        # only need to make client args, no graph args
        app.make_client_args(parser=parser)
