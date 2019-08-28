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
from .recording import record_self
import tensorflow as tf
from contextlib import contextmanager, ExitStack
from common import parse
import tensorflow.contrib.gate as gate
import time
import json
import threading

class LocalRuntime(runtime.Runtime):
    @staticmethod
    def name():
        return "local"

    @staticmethod
    def help_message():
        return "run an application entirely in a local process"

    @classmethod
    def add_arguments(cls, parser):
        cls.add_record_args(parser=parser)
        parser.add_argument("--summary", default=False, action="store_true", help="record a Tensorflow graph summary")
        parser.add_argument("--summary-interval", default=1, type=parse.numeric_min_checker(numeric_type=float,
                                                                                            minimum=0.1,
                                                                                            message="Can't have too small of an interval"),
                            help="interval in seconds for recording summary intervals")

    def _run_filewriter_thread(self, sess, coord, writer, summaries, event, interval):
        interval = float(interval)
        assert isinstance(interval, float)
        global_step = 0
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

    def _run_application(self, ApplicationClass, args):
        device_counts = ApplicationClass.device_counts(args=args)
        ingress_args = ApplicationClass.process_ingress_args(args=args)
        def repeated_fake_device_context():
            @contextmanager
            def fake_device():
                yield
            return fake_device
        devices = {
            k: tuple(repeated_fake_device_context() for _ in range(type_count))
            for k, type_count in device_counts.items()
        }
        application = ApplicationClass(args=args, devices=devices)

        # must do this AFTER app construction because app can modify args (e.g. queue lengths)
        if args.record_args:
            self.write_out_args(args=args)

        results = {}

        with ExitStack() as context_stack:
            full_start = time.time()
            if args.record_stats:
                context_stack.enter_context(cm=record_self(outdir=args.output_directory))
            with tf.Session() as sess:
                init_ops = (tf.local_variables_initializer(), tf.global_variables_initializer())
                sess.run(init_ops)
                run_first = application.run_first
                if len(run_first) > 0:
                    self.log.info("Running {} run_first tensors".format(len(run_first)))
                    sess.run(run_first)
                summary = args.summary
                coord = tf.train.Coordinator()
                threads = []
                queue_runner_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                gate_runner_threads = gate.start_gate_runners(sess=sess, coord=coord)
                credit_runner_threads = gate.start_credit_suppliers(sess=sess, coord=coord)
                threads.extend(queue_runner_threads)
                threads.extend(gate_runner_threads)
                threads.extend(credit_runner_threads)

                if coord.should_stop():
                    raise Exception("Coordinator stopped on initialization. Check for other errors")
                else:
                    self.log.debug("Starting successful")

                if summary:
                    summary_writer = tf.summary.FileWriter(graph=sess.graph, logdir=str(args.output_directory))
                    summary_event = threading.Event()
                    summary_thread = threading.Thread(target=self._run_filewriter_thread, kwargs={
                        "sess": sess,
                        "coord": coord,
                        "writer": summary_writer,
                        "event": summary_event,
                        "interval": args.summary_interval,
                        "summaries": tf.summary.merge_all()
                    })
                    summary_thread.start()

                try:
                    end_start_time = time.time()
                    results = application.run_client_request(client_args=ingress_args, sess=sess)
                    assert results is not None
                except Exception as e:
                    self.log.error("Running client request returned exception: '{}'".format(e))
                else:
                    application.process_egress_results(results=results.results, args=args)
                    startup_time = end_start_time - full_start
                    perf_results = {
                        "client_latency" : results.run_time,
                        "startup_latency": startup_time
                    }
                    outfile = args.output_directory / "results.json"
                    with outfile.open("w") as f:
                        json.dump(obj=perf_results, fp=f, indent=2)
                    self.log.info("Client latency (seconds): {cl}\nFull latency(seconds): {fl}".format(cl=results.run_time, fl=startup_time+results.run_time))
                finally:
                    try:
                        application.stop(sess=sess)
                    except Exception as e:
                        self.log.error("Stopping client runtime raised exception: '{}'".format(e))
                    if summary:
                        summary_event.set()
                        summary_thread.join()
                        summary_writer.flush()

    @staticmethod
    def _populate_app_args(parser, app):
        app.make_graph_args(parser=parser)
        app.make_client_args(parser=parser)
