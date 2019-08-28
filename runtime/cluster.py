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
from . import runtime, dist_common, recording
from common import parse
import Pyro4
import Pyro4.socketutil
import multiprocessing
import time
import enum
import tensorflow as tf
from .worker import SessionState
import tensorflow.contrib.gate as gate
import threading
from contextlib import ExitStack, contextmanager
import concurrent.futures as futures
import random
import re
from collections import namedtuple
import logging
logging.getLogger("Pyro4.core").setLevel(level=logging.DEBUG)

class MasterState(enum.Enum):
    starting = "starting"
    running = "running"
    stopping = "stopping"
    exitted = "exitted"
    error = "error"
    not_yet_started = "not_yet_started"

DeviceAssignment = namedtuple("DeviceAssignment", ("worker", "tf_device", "cluster_endpoint"))

class ClusterRuntime(runtime.Runtime):

    master_job_name = "master"
    worker_job_name = "worker"
    _name_parser = re.compile("(?P<worker_type>[a-zA-Z0-9\-_]+)\.\d+$")

    def __init__(self):
        super().__init__()
        self.executor = None
        self.app_sess_coord = None # tuple so that these can be assigned in one swoop
        self._app_name = None
        recording.recording_cleanup()

    @staticmethod
    def name():
        return "cluster"

    @staticmethod
    def help_message():
        return "run an application on a cluster"

    @classmethod
    def add_arguments(cls, parser):
        cls.add_record_args(parser=parser)
        parser.add_argument("--summary", default=False, action="store_true", help="record a Tensorflow graph summary")
        parser.add_argument("--summary-interval", default=1, type=parse.numeric_min_checker(numeric_type=float,
                                                                                            minimum=0.1,
                                                                                            message="Can't have too small of an interval"),
                            help="interval in seconds for recording summary intervals")

        # related to the timing for the master worker
        parser.add_argument("--master-startup-poll-interval", default=1, type=parse.numeric_min_checker(minimum=0.1, numeric_type=float, message="must have a sensible (>100ms) wait time for startup check"), help="the amount of time to wait when checking for worker status on startup")
        parser.add_argument("--master-shutdown-interval", default=1, type=parse.numeric_min_checker(minimum=0.1, numeric_type=float, message="must have a sensible (>100ms) wait time for startup check"), help="the amount of time to wait when checking for worker status on startup")

        # related to the pyro server which to connect
        parser.add_argument("-n", "--pyro-number", default=random.randint(0, 2**30), type=int, help="number to assign to this server in the naming system")
        parser.add_argument("--pyro-ns-port", type=int, help="override default Pyro4 nameserver port")
        parser.add_argument("--pyro-ns-host", help="override default Pyro4 nameserver port")

    def _get_workers(self, ns_host, ns_port):
        with Pyro4.locateNS(host=ns_host, port=ns_port) as ns:
            prefix = dist_common.pyro_worker_prefix
            prefix_slice = len(prefix) + 1 # the +1 to consume the dot after the prefix name
            for k, worker_uri in ns.list(prefix=dist_common.pyro_worker_prefix).items():
                self.log.info("Found worker at {}".format(worker_uri))
                yield k[prefix_slice:], Pyro4.Proxy(worker_uri)

    def _run_filewriter_thread(self, sess, coord, outdir, event, interval):
        with sess.graph.as_default():
            summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(graph=sess.graph, logdir=str(outdir))
        interval = float(interval)
        global_step = 0
        try:
            if summaries is None:
                self.log.error("SummaryWriter has no summaries!")
                return
            while not (coord.should_stop() or event.is_set()):
                event_results = sess.run(summaries)
                writer.add_summary(summary=event_results,
                                   global_step=global_step)
                time.sleep(interval)
                global_step += 1
            if coord.should_stop():
                self.log.debug("SummaryWriter thread detected stopped coordinator")
            if event.is_set():
                self.log.debug("SummaryWriter thread detected set event")
        except Exception as e:
            self.log.error("SummaryWriter thread got exception '{e}'".format(e=e))
        else:
            self.log.debug("SummaryWriter thread exited normally")
        finally:
            writer.close()

    def _start_workers(self, assignments, cluster_dict, startup_poll_interval):
        meta_graph_def = tf.train.export_meta_graph()
        meta_graph_def_as_string = meta_graph_def.SerializeToString()
        assert isinstance(meta_graph_def_as_string, bytes)
        with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as tpe:
            for job_name, devices_by_index in assignments.items():
                for idx, dev_assignment in devices_by_index.items():
                    tpe.submit(dev_assignment.worker.run,
                               job_name=job_name,
                               task_index=idx,
                               cluster_dict=cluster_dict,
                               graph_def=meta_graph_def_as_string)
        self.log.info("Started all workers. Now will wait for them to fully start...")
        waiting = True
        while waiting:
            waiting = False
            for job_name, devices_by_index in assignments.items():
                for idx, dev_assignment in devices_by_index.items():
                    device = dev_assignment.worker
                    state = SessionState(device.state)
                    if state == SessionState.running:
                        continue
                    elif state == SessionState.starting:
                        self.log.info("Waiting for worker {name}:{idx} to start".format(name=job_name, idx=idx))
                        waiting = True
                    else:
                        raise Exception("Worker {name}:{idx} has bad status: '{status}' on startup".format(idx=idx,
                                                                                                           name=job_name,
                                                                                                           status=state))
            if waiting:
                time.sleep(startup_poll_interval)
        self.log.debug("All workers started. Now serving...")

    def _setup_workers(self, app_name, device_counts, ns_host, ns_port):
        def repeated_device_context(device_name):
            @contextmanager
            def _my_func():
                with tf.device(device_name):
                    yield
            return _my_func

        all_workers = dict(self._get_workers(ns_host=ns_host,
                                             ns_port=ns_port))
        workers_by_type = {}
        for worker_name, worker in all_workers.items():
            match = self._name_parser.search(worker_name)
            if match is None:
                continue
            name = match.group("worker_type")
            if name not in workers_by_type:
                workers_by_type[name] = [worker]
            else:
                workers_by_type[name].append(worker)
        assignments = {}
        port_mappings_by_host = {}
        for device_key, num_requested in device_counts.items():
            if num_requested == 0:
                continue
            if device_key not in workers_by_type:
                raise Exception("Application '{app_name}' requested workers of type '{t}', but none are registered".format(t=device_key,
                                                                                                                           app_name=app_name))
            available_workers = workers_by_type[device_key]
            if len(available_workers) < num_requested:
                raise Exception("Application '{app_name}' requested {req} workers of type '{t}', but only found {found}".format(
                    app_name=app_name, req=num_requested, t=device_key, found=len(available_workers)
                ))
            chosen_workers = available_workers[:num_requested]
            devices_by_idx = {}
            for idx, worker in enumerate(chosen_workers):
                device_info = worker.get_tf_dist_info(current_reservations=port_mappings_by_host)
                host = device_info["host"]
                port = device_info["port"]
                if host not in port_mappings_by_host:
                    port_mappings_by_host[host] = { port }
                else:
                    assert port not in port_mappings_by_host[host], "Worker return port that was already in the mapping!"
                    port_mappings_by_host[host].add(port)
                cluster_info_string = "{h}:{p}".format(h=host, p=port)
                devices_by_idx[idx] = DeviceAssignment(worker=worker,
                                                       cluster_endpoint=cluster_info_string,
                                                       tf_device=repeated_device_context(dist_common.make_tf_device_name(
                                                           job_name=device_key,
                                                           task_index=idx
                                                       )))
            assignments[device_key] = devices_by_idx
        return assignments, port_mappings_by_host

    def _start_pyro(self, ns_host, ns_port, pyro_number):
        daemon = Pyro4.Daemon(host=Pyro4.socketutil.getIpAddress(None, workaround127=True))
        master_uri = daemon.register(self)
        self.log.debug("master uri: {}".format(master_uri))
        with Pyro4.locateNS(host=ns_host,
                            port=ns_port) as ns:
            ns.register(":".join((dist_common.pyro_master_name, str(pyro_number))), master_uri)
        self.log.info("Registered Pyro4 daemon: {}".format(master_uri))
        return daemon

    def _construct_application(self, assignments, port_mappings_by_host,
                               ApplicationClass, args):
        master_device_name = dist_common.make_tf_device_name(
            job_name=self.master_job_name, task_index=0
        )
        master_info = dist_common.get_tf_dist_info(current_reservations=port_mappings_by_host)
        cluster_dict = { self.master_job_name: {0: ":".join((master_info["host"], str(master_info["port"])))}}
        cluster_dict.update((job_key, { idx: dev_assignment.cluster_endpoint for idx, dev_assignment in v.items() })
                            for job_key, v in assignments.items())
        with tf.device(master_device_name):
            devices = {
                job_key: tuple(da.tf_device for da in v.values())
                for job_key, v in assignments.items()
            }
            application = ApplicationClass(args=args, devices=devices)
        return cluster_dict, application, master_device_name

    def _run_master(self, sess, coord, application, master_device_name):
        init_ops = (tf.local_variables_initializer(), tf.global_variables_initializer())
        tf.report_uninitialized_variables()
        sess.run(init_ops)
        run_first = application.run_first
        if len(run_first) > 0:
            self.log.warning("App has {} run_first tensors, but can't run them across sessions".format(len(run_first)))
        threads = []
        queue_runner_threads = tf.train.start_queue_runners(sess=sess, coord=coord, device=master_device_name)
        self.log.info("Queue runners ({device}): {ths}".format(device=master_device_name, ths=", ".join(t.name for t in queue_runner_threads)))
        gate_runner_threads = gate.start_gate_runners(sess=sess, coord=coord, device=master_device_name)
        self.log.info("Gate runners ({device}): {ths}".format(device=master_device_name, ths=", ".join(t.name for t in gate_runner_threads)))
        credit_runner_threads = gate.start_credit_suppliers(sess=sess, coord=coord, device=master_device_name)
        self.log.info("Credit runners ({device}): {ths}".format(device=master_device_name, ths=", ".join(t.name for t in credit_runner_threads)))
        threads.extend(queue_runner_threads)
        threads.extend(gate_runner_threads)
        threads.extend(credit_runner_threads)

        time.sleep(1)
        if coord.should_stop():
            raise Exception("Coordinator stopped on initialization. Check for other errors!")
        else:
            self.log.debug("Starting successful")

        self.app_sess_coord = (application, sess, coord)
        coord.wait_for_stop()

        # TODO not sure if I need this
        coord.raise_requested_exception()

    def _run_application(self, ApplicationClass, args):
        assert self.executor is None, "Cluster is restarting somehow!"
        self.executor = futures.ThreadPoolExecutor(max_workers=args.max_parallel_clients)
        app_name = ApplicationClass.name()
        self._app_name = app_name

        device_counts = ApplicationClass.device_counts(args=args)

        assignments, port_mappings_by_host = self._setup_workers(app_name=app_name,
                                                                 device_counts=device_counts,
                                                                 ns_host=args.pyro_ns_host,
                                                                 ns_port=args.pyro_ns_port)

        cluster_dict, application, master_device_name = self._construct_application(assignments=assignments,
                                                                                    port_mappings_by_host=port_mappings_by_host,
                                                                                    ApplicationClass=ApplicationClass, args=args)

        # must do this AFTER app construction because app can modify args (e.g. queue lengths)
        if args.record_args:
            self.write_out_args(args=args)

        daemon = self._start_pyro(
            ns_host=args.pyro_ns_host,
            ns_port=args.pyro_ns_port,
            pyro_number=args.pyro_number
        )
        daemon_thread = threading.Thread(target=daemon.requestLoop, name="pyro4_master_daemon")

        cluster_spec = tf.train.ClusterSpec(cluster=cluster_dict)
        server = tf.train.Server(server_or_cluster_def=cluster_spec,
                                 job_name=self.master_job_name, task_index=0)

        coord = tf.train.Coordinator()
        try:
            daemon_thread.start()
            time.sleep(1.25) # make sure that the master is up and ready to accept requests
            with ExitStack() as context_stack:
                sess = context_stack.enter_context(tf.Session(target=server.target))
                post_session_sleep_time = 5
                self.log.info("Waiting {s} seconds for master Session to fully start...".format(s=post_session_sleep_time))
                time.sleep(post_session_sleep_time)
                self.log.info("Done waiting for master Session start")
                self._start_workers(
                    assignments=assignments,
                    cluster_dict=cluster_dict,
                    startup_poll_interval=args.master_startup_poll_interval
                )

                if args.record_stats:
                    context_stack.enter_context(cm=recording.record_self(outdir=args.output_directory))

                if args.summary:
                    summary_event = threading.Event()
                    summary_thread = threading.Thread(target=self._run_filewriter_thread, kwargs={
                        "sess": sess,
                        "coord": coord,
                        "outdir": args.output_directory,
                        "event": summary_event,
                        "interval": args.summary_interval
                    })
                    summary_thread.start()

                try:
                    self._run_master(
                        sess=sess,
                        coord=coord,
                        master_device_name=master_device_name,
                        application=application
                    )
                finally:
                    if args.summary:
                        summary_event.set()

                    self.log.debug("Attempting application.stop()...")
                    try:
                        application.stop(sess=sess)
                        sess.close()
                    except Exception as e:
                        self.log.warning("Ignoring exception '{e}' thrown by Application.stop".format(e=e))
                    else:
                        self.log.debug("Successfully ran application.stop()")
        except Exception as e:
            self.log.error("Master shutting down due to exception: {e}".format(e=e))
            raise e
        else:
            self.log.debug("Master shutting down normally.")
        finally:
            coord.request_stop()
            self.app_sess_coord = None
            wait_timeout = 60
            if coord.wait_for_stop(timeout=wait_timeout):
                self.log.debug("coord stop successful")
            else:
                self.log.error("Couldn't stop master thread after {} seconds of timeout".format(wait_timeout))

            # will definitely be available, based on construction. No need to check.
            self.executor.shutdown()
            self.executor = None

            # Tell the workers to shut down
            timeout=10
            with futures.ThreadPoolExecutor(max_workers=8) as pool:
                for job_name, devices_by_index in assignments.items():
                    for idx, dev_assignment in devices_by_index.items():
                        worker = dev_assignment.worker
                        pool.submit(worker.stop_and_reset, timeout=timeout)

            # Finally, shut down the daemon. Do this last if workers call back into this
            daemon.shutdown()
            daemon_join_timeout = 10
            daemon_thread.join(timeout=daemon_join_timeout)
            if daemon_thread.is_alive():
                self.log.error("Daemon thread still alive after timeout of {} seconds".format(daemon_join_timeout))

    @staticmethod
    def _populate_app_args(parser, app):
        # only graph args, no client args
        app.make_graph_args(parser=parser)

    @Pyro4.expose
    def run_client_request(self, ingress_args):
        def run_request():
            a = self.app_sess_coord
            if a is None:
                raise Exception("Can't run client request. Application is not running. App/Sess/Coord is None.")
            app, sess, coord = a
            assert app is not None and sess is not None
            with sess.as_default():
                results = app.run_client_request(client_args=ingress_args,
                                                 sess=sess)
                return {
                    dist_common.results_key: results.results,

                    # these return the actual UNIX time bounds for when this got a slot and was submitted
                    "start_time": results.start_time,
                    "end_time": results.end_time,

                    # run_time is just start_time - end_time
                    # wait_time is the time BEFORE start_time that this request waited for a slot
                    "run_time": results.run_time,
                    "wait_time": results.wait_time,
                }
        if self.executor is None or self.app_sess_coord is None:
            raise Exception("Application is not running.{e}{s}".format(
                e="" if self.executor is None else " Executor is None.",
                s="" if self.app_sess_coord is None else " Session is None."
            ))
        assert isinstance(self.executor, futures.Executor)
        result = self.executor.submit(run_request)
        return result.result()

    @Pyro4.expose
    def kill(self):
        if self.app_sess_coord is not None:
            app, sess, coord = self.app_sess_coord
            coord.request_stop()

    @Pyro4.expose
    @property
    def app_name(self):
        return self._app_name
