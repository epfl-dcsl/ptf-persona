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
from . import dist_common
import Pyro4
import Pyro4.socketutil
import random
from common import parse
import tensorflow as tf
import tensorflow.contrib.gate as gate
import threading
import logging
import shutil
logging.basicConfig(level=logging.DEBUG)
import enum
import time
from serpent import tobytes
from tensorflow.core.protobuf import meta_graph_pb2
from contextlib import ExitStack
from .recording import record_self, recording_cleanup

log = logging.getLogger(name=__file__)
log.setLevel(level=logging.DEBUG)
logging.getLogger("Pyro4.core").setLevel(level=logging.DEBUG)

worker_command = "worker"

class SessionState(enum.Enum):
    not_yet_started = "not_yet_started"
    starting = "starting"
    running = "running"
    stopping = "stopping"
    shut_down = "shut_down"
    error = "error"

class Session:
    """
    Should be launched in a separate thread, the run() method
    """
    def __init__(self, graph_def, cluster_dict, job_name, task_index, record_directory, pyro_host, pyro_port):
        self.job_name = job_name
        self.task_index = task_index
        self.log = logging.getLogger("|".join((self.__class__.__name__, job_name, str(task_index))))
        self.log.setLevel(level=logging.DEBUG)
        self.device_name = dist_common.make_tf_device_name(job_name=job_name,
                                                           task_index=task_index)
        self.event = threading.Event()
        self.cluster_dict = cluster_dict
        self.cluster_spec = tf.train.ClusterSpec(cluster_dict)
        self.graph_def = graph_def
        self.server = tf.train.Server(server_or_cluster_def=self.cluster_spec,
                                      job_name=job_name, task_index=task_index)
        self.run_thread = None
        self._state = SessionState.not_yet_started
        self.record_directory = record_directory

        self.pyro_host = pyro_host
        self.pyro_port = pyro_port

    def await(self, timeout):
        if self.run_thread is not None:
            self.event.set()
            # TODO probably need to close here
            self.log.debug("Waiting for run_thread to stop for {s} seconds".format(s=timeout))
            self.run_thread.join(timeout=timeout)
            if self.run_thread.is_alive():
                raise Exception("run_thread still running after {s} seconds timeout".format(s=timeout))
            self.run_thread = None
            self.log.debug("run_thread stopped successfully")
        elif self.event.is_set():
            self.log.debug("await() requested, but run_thread already stopped")
        else:
            self.log.debug("await() requested, but Session never started")

    @property
    def state(self):
        return self._state

    def _stop_master(self):
        try:
            with Pyro4.locateNS(host=self.pyro_host,
                                port=self.pyro_port) as ns:
                masters = ns.list(prefix=dist_common.pyro_master_name)
            num_masters = len(masters)
            if num_masters != 1:
                self.log.error("Found {n} masters when attempting to shut down worker. Avoiding master call!".format(n=num_masters))
            master = Pyro4.Proxy(tuple(masters.values())[0])
            master.kill()
        except Exception as e:
            log.warning("Ignoring the following exception when running _stop_master: {e}".format(e=e))

    def run(self, run_sleep_interval, job_name, task_index, startup_sleep):
        device_name = dist_common.make_tf_device_name(job_name=job_name,
                                                      task_index=task_index)
        def _run():
            try:
                self._state = SessionState.starting
                tf.reset_default_graph()
                tf.train.import_meta_graph(self.graph_def)
                with tf.Session(target=self.server.target) as sess:
                    with ExitStack() as context_stack:
                        self.log.info("Waiting {} seconds after session created...".format(startup_sleep))
                        time.sleep(startup_sleep)
                        self.log.info("Done waiting for session startup")

                        init_ops = (tf.local_variables_initializer(), tf.global_variables_initializer())
                        sess.run(init_ops)

                        if self.record_directory is not None:
                            # kill self is necessary in case the worker restarts
                            context_stack.enter_context(record_self(outdir=self.record_directory, kill_self=True))

                        coord = tf.train.Coordinator()
                        threads = []
                        queue_runner_threads = tf.train.start_queue_runners(sess=sess, coord=coord, device=device_name)
                        self.log.info("Queue runners ({device}): {ths}".format(device=device_name, ths=", ".join(t.name for t in queue_runner_threads)))
                        gate_runner_threads = gate.start_gate_runners(sess=sess, coord=coord, device=device_name)
                        self.log.info("Gate runners ({device}): {ths}".format(device=device_name, ths=", ".join(t.name for t in gate_runner_threads)))
                        credit_runner_threads = gate.start_credit_suppliers(sess=sess, coord=coord, device=device_name)
                        self.log.info("Credit runners ({device}): {ths}".format(device=device_name, ths=", ".join(t.name for t in credit_runner_threads)))
                        threads.extend(queue_runner_threads)
                        threads.extend(gate_runner_threads)
                        threads.extend(credit_runner_threads)
                        self._state = SessionState.running
                        while (not (coord.should_stop() or self.event.is_set())) or not any(t.is_alive() for t in threads):
                            time.sleep(run_sleep_interval)
                        self.log.info("Exited run loop. Now stopping")
                        self._state = SessionState.stopping
                        self.event.set()
                        if not coord.should_stop():
                            if not self.event.is_set():
                                self.log.error("coord shouldn't stop and event isn't set!")
                            self.log.debug("requesting coord to stop")
                            # coord.request_stop(ex=Exception("Shutting down worker"))
                            coord.request_stop()
                            self.log.debug("requested coord stop")
                        assert coord.should_stop() and self.event.is_set()
                    # self.log.debug("Running sess.close()")
                    # sess.close()
                    # self.log.debug("Ran sess.close()")
                self.log.debug("Session is done")
            except Exception as e:
                log.error("Got exception: {e}".format(e=e))
                self._state = SessionState.error
                raise e
            else:
                self._state = SessionState.shut_down
            finally:
                self._stop_master()
        if self.run_thread is not None:
            raise Exception("Attempting to restart a session")
        self.run_thread = threading.Thread(target=_run, name="Session_main_thread")
        self.run_thread.start()

class Worker:
    def __init__(self, args):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(level=logging.DEBUG)
        self.run_sleep_interval = args.run_sleep_interval
        self._lock = threading.Lock()
        self.session = None
        self.outdir = None
        self.args = args

    @staticmethod
    @Pyro4.expose
    def get_tf_dist_info(current_reservations, start_port=30000):
        return dist_common.get_tf_dist_info(start_port=start_port, current_reservations=current_reservations)

    @Pyro4.expose
    def run(self, graph_def, cluster_dict, job_name, task_index):
        graph_def = tobytes(graph_def) # for remote calls, to undo the serpent deserializer
        full_graph_def = meta_graph_pb2.MetaGraphDef.FromString(graph_def)
        with self._lock: # because this could be reentrant
            self.log.info("Attempting to start job {name}, task index {ti}".format(
                name=job_name, ti=task_index
            ))
            if self.session is not None:
                self.log.warning("Attempting to assign new slave session when one already exists! Stopping old one")
                self.stop_and_reset()

            # Note: outdir will be absolute because of the argument parser
            if len(job_name) == 0:
                outdir = self.args.output_directory / "worker.{}".format(task_index)
            else:
                outdir = self.args.output_directory / "{n}.{i}".format(n=job_name, i=task_index)
            outdir = outdir.absolute()

            if self.args.record_stats:
                if outdir.exists():
                    shutil.rmtree(str(outdir))
                outdir.mkdir()

            self.session = Session(graph_def=full_graph_def,
                                   cluster_dict=cluster_dict,
                                   job_name=job_name,
                                   task_index=task_index,
                                   pyro_host=self.args.pyro_ns_host,
                                   pyro_port=self.args.pyro_ns_port,
                                   record_directory=outdir if self.args.record_stats else None)
            try:
                self.session.run(run_sleep_interval=self.run_sleep_interval, job_name=job_name, task_index=task_index,
                                 startup_sleep=self.args.startup_sleep)
            except:
                self.session = None # failed to launch
                raise
            else:
                if self.args.record_stats:
                    self.outdir = outdir

    @Pyro4.expose
    @property
    def state(self):
        if self.session is None:
            return SessionState.not_yet_started
        else:
            return self.session.state

    @Pyro4.expose
    @property
    def output_directory(self):
        return str(self.outdir)

    @Pyro4.expose
    def stop_and_reset(self, timeout=10):
        with self._lock:
            if self.session is not None:
                if self.session.state not in {SessionState.shut_down, SessionState.error}:
                    self.log.debug("Awaiting session with state '{state}'".format(state=self.session.state))
                    self.session.await(timeout=timeout)
                self.session = None
            else:
                self.log.info("stop_and_reset() attempted when no Session is set")

def add_args(parser):
    parser.add_argument("--record-stats", default=False, action="store_true", help="store statistics for this process into the output directory")
    parser.add_argument("-o", "--output-directory", default=".", type=parse.path_exists_checker(check_dir=True), help="path in which to store the directory of outputs")
    parser.add_argument("-n", "--number", default=random.randint(0, 2**30), type=int, help="number to assign to this server in the naming system")
    parser.add_argument("--safe-register", default=False, action="store_true", help="error if the name already exists in the name server")
    parser.add_argument("--pyro-ns-port", type=int, help="override default Pyro4 nameserver port")
    parser.add_argument("--pyro-ns-host", help="override default Pyro4 nameserver port")
    parser.add_argument("-i", "--run-sleep-interval", dest="run_sleep_interval", default=2, type=parse.numeric_min_checker(0.5, numeric_type=float, message="must wait at least 0.5 seconds"),
                        help="number of seconds to sleep while in the run loop")
    parser.add_argument("-w", "--worker-name", default="", help="if set, use this exact name to register on the nameserver. An error will occur if this name is already taken")
    parser.add_argument("--startup-sleep", default=3, type=parse.numeric_min_checker(numeric_type=float, minimum=1, message="must wait at least 1 second after worker starts"),
                        help="number of seconds to sleep after session is initialized")

def run(args):
    recording_cleanup()
    daemon = Pyro4.Daemon(host=Pyro4.socketutil.getIpAddress(None, workaround127=True))
    worker = Worker(args=args)
    worker_uri = daemon.register(worker)
    log.info("Worker uri: {}".format(worker_uri))
    ns_name = ".".join((dist_common.pyro_worker_prefix, args.worker_name, str(args.number)))
    with Pyro4.locateNS(host=args.pyro_ns_host, port=args.pyro_ns_port) as ns:
        ns.register(name=ns_name, uri=worker_uri, safe=args.safe_register)
    log.info("Registered worker daemon: {}".format(worker_uri))
    daemon.requestLoop()
    log.info("daemon request exiting. stopping worker")
    worker.stop_and_reset()
    log.info("worker stopped")

    # Pyro4.Daemon.serveSimple(
    #     {
    #         Worker(args=args): ".".join((dist_common.pyro_worker_prefix, str(args.number)))
    #     }
    # )
