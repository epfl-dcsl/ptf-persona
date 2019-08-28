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
import threading
from collections import namedtuple
from contextlib import contextmanager
from common.parse import numeric_min_checker
import time
import logging

class Application(abc.ABC):

    ClientSlot = namedtuple("ClientSlot", ("ingress_placeholders", "egress_dequeue"))
    TimedResults = namedtuple("TimedResults", ("results", "wait_time", "run_time",
                                               "start_time", "end_time"))

    # All subclasses should have an __init__ method that takes the args returned by make_args
    def __init__(self, args, devices, log_level=logging.DEBUG):
        """
        This init call will construct the graph for this application, using devices as necessary
        :param args:
        :param devices: a list of devices
        """
        self._run_first = []
        self._slot_cv = threading.Condition()
        self.log = logging.getLogger(name=self.name())
        self.log.setLevel(level=log_level)
        self._max_clients = args.max_parallel_clients

        self.check_devices(
            request_map=self.device_counts(args=args),
            device_map=devices
        )

        # Note: this must go last so that the above things are all ready
        self._slots = list(self._construct_graph(args=args, device_map=devices, num_client_slots=args.max_parallel_clients))

    @staticmethod
    @abc.abstractmethod
    def name():
        pass

    @staticmethod
    def help_message():
        return ""

    @classmethod
    @abc.abstractmethod
    def _make_graph_args(cls, parser):
        pass

    @classmethod
    def make_graph_args(cls, parser):
        cls._make_graph_args(parser=parser)
        parser.add_argument("--max-parallel-clients", default=8, type=numeric_min_checker(1, "must allow at least one parallel client"), help="number of parallel clients this App should allow")

    @classmethod
    @abc.abstractmethod
    def device_counts(cls, args):
        """
        :param args:
        :return: a map of {device_name: count}, where device_name is a string and count is a integer
        """
        return {}

    @classmethod
    def check_devices(cls, device_map, request_map):
        for device_type, count in request_map.items():
            if count == 0:
                continue
            if device_type not in device_map:
                raise Exception("Application '{app_name}' expected {count} devices for type '{name}', but wasn't given any".format(
                    count=count, name=device_type, app_name=cls.name()))
            devices = device_map[device_type]
            device_count = len(devices)
            if device_count < count:
                raise Exception("Application '{app_name}' expected {count} devices for type '{name}', but only got {actual}".format(
                    app_name=cls.name(), count=count, actual=device_count, name=device_type
                ))

    @abc.abstractmethod
    def _construct_graph(self, args, device_map, num_client_slots):
        """
        Constructs the graph. Should only be called by Application.__init__

        :param args:
        :return: a list of len() == parallel clients of ClientSlots, for the app to add to its available slots
        """
        pass

    def create_threads(self, sess, coord):
        """
        Create custom threads for this, such as for the ingress and egress processing
        :param sess:
        :param coord:
        :return: a list of threads to be run
        """
        return ()

    @contextmanager
    def client_slot(self):
        with self._slot_cv:
            self._slot_cv.wait_for(lambda : len(self._slots) > 0)
            slot = self._slots.pop()
        # note no try/finally here to protect. if this context manager crashes, we need to kill everything
        # because there's a bug in the logic
        yield slot
        with self._slot_cv:
            self._slots.append(slot)
            self._slot_cv.notify()

    @classmethod
    @abc.abstractmethod
    def make_client_args(cls, parser):
        """
        Add arguments to this parser for running a single "input" instance for this application.

        :param parser:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def process_ingress_args(cls, args):
        """
        Take the args from this and return a dictionary of processed arguments
        :param args: resulting args which have at least the arguments from the associated make_client_args() call
        :return: a dictionary of arguments to give to run_client_request
        """
        pass

    @classmethod
    @abc.abstractmethod
    def process_egress_results(cls, results, args):
        """
        Take the resulting dictionary of results, resulting from the pipeline, and do some final processing with it.
        :param results: the dictionary of results
        :return:
        """
        pass

    @abc.abstractmethod
    def _run_client_request(self, client_args, client_slot, sess):
        """
        Actually run the client request through the pipeline
        :param client_args: a dictionary as constructed by process_ingress_args
        :param client_slot:
        :return: a dictionary to be processed by process_egress_results
        """
        pass

    def run_client_request(self, client_args, sess):
        pre_start_time = time.time()
        with self.client_slot() as slot:
            actual_start_time = time.time()
            results = self._run_client_request(client_args=client_args,
                                               client_slot=slot,
                                               sess=sess)
            actual_stop_time = time.time()
        if results is None:
            self.log.warning("_run_client_request returned None. This may be a bug!")
        wait_time = max(0.0, actual_start_time-pre_start_time)
        run_time = actual_stop_time - actual_start_time
        return self.TimedResults(results=results, wait_time=wait_time, run_time=run_time,
                                 start_time=actual_start_time, end_time=actual_stop_time)

    @abc.abstractmethod
    def stop(self, sess):
        pass

    @property
    def run_first(self):
        return self._run_first

    def _add_run_first(self, tensor):
        self._run_first.append(tensor)
