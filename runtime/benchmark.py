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
from .client import ClientRuntime
from .runtime import CustomArgEncoder
from common import parse, util
import Pyro4
import Pyro4.naming, Pyro4.socketutil
import time
import argparse
import pprint
import itertools
import shlex
import spur
from scp import SCPClient
import subprocess
import pathlib
import abc
from argparse import ArgumentTypeError
import json
from copy import deepcopy
import threading
from contextlib import contextmanager
from functools import partial
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
import app
from app.app import Application
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)
log.setLevel(level=logging.DEBUG)

# Keep this here! the ./persona script uses it
benchmark_command = "bench"
min_runtime = 0.1

file_logging_formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s: %(message)s")
stream_formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s: %(message)s")

class WrapperWriter:
    # redirect used to only write output when newline is received at the end

    def __init__(self, prefix, output_directory):
        local_log = logging.Logger(name=prefix, level=logging.DEBUG)
        assert not local_log.hasHandlers()
        local_log.setLevel(level=logging.DEBUG)
        if output_directory is not None:
            assert isinstance(output_directory, pathlib.Path) and output_directory.exists() and output_directory.is_dir(), "expected output_directory to be a pathlib.Path"
            outfile = output_directory / "{}.txt".format(prefix)
            if outfile.exists():
                outfile.unlink()
            file_output = logging.FileHandler(filename=str(outfile.absolute()), mode="w")
            file_output.setFormatter(file_logging_formatter)
            local_log.addHandler(file_output)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        local_log.addHandler(stream_handler)
        self.log = local_log
        self.buff = ""

    def write(self, output):
        out = output
        if isinstance(out, bytes):
            out = out.decode()
        self.buff += out
        split_index = self.buff.find("\n")
        if split_index >= 0: # and not self.stop_event.is_set():
            if split_index > 0:
                self.log.info(self.buff[:split_index])
            self.buff = self.buff[split_index+1:] # to skip over the newline

class LocalLogger:

    # provides a local logging service to log to a file

    def __init__(self, log_name, logging_directory, level=logging.DEBUG):
        log = logging.Logger(name=log_name)
        log.setLevel(level=level)
        output_file = logging_directory / "{}.txt".format(log_name)
        file_output = logging.FileHandler(filename=str(output_file.absolute()), mode="w")
        file_output.setFormatter(file_logging_formatter)
        stream_output = logging.StreamHandler()
        stream_output.setFormatter(stream_formatter)
        log.addHandler(stream_output)
        log.addHandler(file_output)
        self.log = log

@contextmanager
def run_pyro_ns():
    ip_addr = Pyro4.socketutil.getIpAddress(None, workaround127=True)
    uri, daemon, broadcast_server = Pyro4.naming.startNS(host=ip_addr)
    assert broadcast_server is not None, "a broadcast server should be created"
    stop_signal = threading.Event()
    ns_thread = threading.Thread(target=daemon.requestLoop, kwargs={"loopCondition": lambda : not stop_signal.is_set() }, daemon=True, name="ns_thread")
    ns_thread.start()
    svr_ip_addr, port = daemon.sock.getsockname()

    try:
        yield svr_ip_addr, port
    finally:
        # this closes at the end of the experiment, so no use in actually waiting for it to stop (hence daemon=True)
        stop_signal.set()
        daemon.close()
        broadcast_server.close()

class RemoteWorker(LocalLogger, abc.ABC):
    def __init__(self, host, username, tensorflow_path, shell_path, output_directory, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.username = username
        assert isinstance(output_directory, pathlib.Path)
        assert output_directory.exists() and output_directory.is_dir(), "output directory {} doesn't exist or isn't a directory".format(str(output_directory))
        self.out_dir = output_directory

        # need to be PurePath instances because the filesystems are remote
        self.tensorflow_path = pathlib.PurePath(tensorflow_path)
        self.shell_path = pathlib.PurePath(shell_path)
        self.shell = spur.SshShell(hostname=host,
                                   username=username,
                                   missing_host_key=spur.ssh.MissingHostKey.accept)
        if self.shell.run(("ls", str(self.tensorflow_path)), allow_error=True).return_code != 0:
            raise Exception("Worker {cls} on host {h} can't find tensorflow directory at path {pth}".format(
                cls=self.__class__.__name__, h=host, pth=str(self.tensorflow_path)
            ))
        activate_path = self.tensorflow_path / "python_dev/bin/activate"
        if self.shell.run(("ls", str(activate_path)), allow_error=True).return_code != 0:
            raise Exception("Worker {cls} on host {h} has tensorflow directory, but not python_dev at path {pth}. May need to run setup_dev.sh".format(
                cls=self.__class__.__name__, h=host, pth=str(activate_path)
            ))
        self.activate_path = activate_path

        if self.shell.run(("ls", str(self.shell_path/"persona")), allow_error=True).return_code != 0:
            raise Exception("Worker {w} on host {h} can't find valid shell path at {pth}".format(
                w=self.__class__.__name__, h=host, pth=str(self.shell_path)
            ))

        string_cmd = """pgrep -u whitlock -fa 'python3 persona' | awk '{print $1}' | xargs kill"""
        wipe_cmd = self.shell.run(("bash", "-c", string_cmd), allow_error=True)
        if wipe_cmd.return_code != 0:
            self.log.info("Trying to wipe existing persona invocations returned code {}".format(wipe_cmd.return_code))

    def spawn_persona(self, command, name=None, clean_csv=False):
        kwargs = {}
        if name is not None:
            kwargs["stdout"] = WrapperWriter(output_directory=self.out_dir, prefix="{name}_stdout".format(name=name))
            kwargs["stderr"] = WrapperWriter(output_directory=self.out_dir, prefix="{name}_stderr".format(name=name))
            kwargs["encoding"] = "utf-8"

        if clean_csv:
            self.log.debug("Cleaning remote CSV")
            cmd = "rm {shell_path}/*.csv".format(shell_path=self.shell_path)
            self.shell.run(("bash", "-c", cmd), allow_error=True)

        cmd = shlex.split("""{tf_activate_path} python3 persona {cmd}""".format(
            cmd=command,
            shell_path=str(self.shell_path),
            tf_activate_path=str(self.activate_path)
        ))
        cmd = ( str(self.shell_path / "run_in_env.sh"), ) + tuple(cmd)

        return self.shell.spawn(
            command=cmd,
            cwd=str(self.shell_path),
            store_pid=True,
            **kwargs)

    def copy_directory_from_remote(self, local_dest_path, remote_source_path):
        SCPClient(transport=self.shell._get_ssh_transport()).get(
            local_path=str(local_dest_path), remote_path=str(remote_source_path), recursive=True)

    @abc.abstractmethod
    def hard_cleanup(self):
        raise NotImplemented

    # add the arguments we expect from the json dict entry
    @classmethod
    def add_kwargs(cls, json_dict):
        for arg_key in ("host", "tensorflow_path", "shell_path"):
            if arg_key not in json_dict:
                raise ArgumentTypeError("RemoteWorker missing argument key '{k}' from dict {d}".format(
                    k=arg_key, d=pprint.pformat(json_dict)
                ))
            yield arg_key, json_dict[arg_key]

class Master(RemoteWorker):
    output_directory_name = "cluster_output"

    def hard_cleanup(self):
        self.shell.run(
            shlex.split("""pkill -f "persona cluster" -u whitlock"""), allow_error=True
        )

    def __init__(self, app_name, app_args, num_clients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_name = app_name
        self.app_args = app_args
        self.num_clients = num_clients

    @classmethod
    def fromJSON(cls, json_dict, **kwargs):
        kwargs.update(cls.add_kwargs(json_dict=json_dict))

        app_name_key = "application"
        if app_name_key not in json_dict:
            raise ArgumentTypeError("Master json dict didn't contain required key '{k}': {d}".format(
                k=app_name_key, d=pprint.pformat(json_dict)
            ))
        kwargs["app_name"] = json_dict[app_name_key]

        app_args_key = "arguments"
        if app_args_key not in json_dict:
            raise ArgumentTypeError("Master json dict didn't contain required key '{k}': {d}".format(
                k=app_args_key, d=pprint.pformat(json_dict)
            ))
        kwargs["app_args"] = json_dict[app_args_key]
        num_clients = json_dict["num_clients"]
        if num_clients < 1:
            raise Exception("Master got number of parallel clients = {}. Need >1 parallel clients!".format(num_clients))
        kwargs["num_clients"] = num_clients

        return cls(**kwargs)

    def run(self, pyro_host, pyro_port):
        def gen_app_args():
            for k,v in self.app_args.items():
                yield "--{k} {v}".format(
                    v=v, k=k.replace("_","-")
                )
            yield "--max-parallel-clients {}".format(self.num_clients)

        command = "cluster {cluster_args} {app_name} {app_args}".format(
            cluster_args="--summary --summary-interval {summary_interval} --record-stats --record-args --output-directory {out_name} --pyro-ns-host {h} --pyro-ns-port {p}".format(
                h=pyro_host, p=pyro_port, summary_interval=1.0,
                out_name=self.output_directory_name),
            app_name=self.app_name,
            app_args=" ".join(gen_app_args())
        )
        return self.spawn_persona(command=command, name="master")

    def copy_result_folder(self, local_dest):
        assert isinstance(local_dest, pathlib.Path), "local_dest must be a pathlib instance"
        self.copy_directory_from_remote(local_dest_path=local_dest,
                                        remote_source_path=self.shell_path / self.output_directory_name)

class Worker(RemoteWorker):

    def __init__(self, worker_name, worker_number, startup_delay, startup_sleep_interval, sleep_interval, **kwargs):
        super().__init__(**kwargs)
        self.worker_name = worker_name
        self.worker_number = worker_number
        self.startup_delay = startup_delay
        self.sleep_interval = sleep_interval
        self.startup_sleep_interval = startup_sleep_interval
        self.remote_output_dir = None

    def hard_cleanup(self):
        self.shell.run(
            shlex.split("""pkill -f "persona worker" -u whitlock"""), allow_error=True
        )

    @classmethod
    def fromJSON(cls, json_dict, **kwargs):
        for desired_key in ("worker_name", "worker_number"):
            if desired_key not in json_dict:
                raise ArgumentTypeError("Worker json dict doesn't have needed key '{k}'".format(k=desired_key))
            kwargs[desired_key] = json_dict[desired_key]

        kwargs.update(cls.add_kwargs(json_dict=json_dict))
        kwargs["log_name"] = "_".join((kwargs["worker_name"], str(kwargs["worker_number"])))
        return cls(**kwargs)

    def run(self, pyro_host, pyro_port):
        command = "worker {worker_args}".format(
            worker_args="--run-sleep-interval {si} --record-stats --safe-register --pyro-ns-host {h} --pyro-ns-port {p} --worker-name {w} --number {num}".format(
                h=pyro_host, p=pyro_port, w=self.worker_name, num=self.worker_number, si=self.sleep_interval)
        )
        return self.spawn_persona(clean_csv=True, command=command, name="worker:{n}:{num}:{h}".format(h=self.host, n=self.worker_name,
                                                                                      num=self.worker_number))

    def populate_remote_output_dir(self, ns):
        prefix = ".".join((dist_common.pyro_worker_prefix, self.worker_name, str(self.worker_number)))
        available = ns.list(prefix=prefix)
        if prefix not in available:
            raise Exception("Searching for worker '{name}', but only found other prefix matches: [{others}]".format(
                name=prefix, others=", ".join(available.keys())
            ))
        worker = Pyro4.Proxy(available[prefix])
        self.remote_output_dir = worker.output_directory

    def copy_csvs(self, local_directory):
        cmd = """scp -o StrictHostKeyChecking=no {remote_host}:{host_path}/*.csv {local_dir}""".format(
            remote_host=self.host, host_path=self.shell_path, local_dir=local_directory
        )
        result = subprocess.run(args=cmd, shell=True)
        if result.returncode != 0:
            self.log.warning("""Attempting to retrieve CSV files with the following command failed with return code {ret}: "{cmd}" """.format(
                ret=result.returncode, cmd=cmd
            ))

    def copy_result_folder(self, local_dest):
        name = "{}.{}".format(self.worker_name, self.worker_number)
        assert self.remote_output_dir is not None, "{name}: output directory not yet populated".format(name=name)
        self.copy_directory_from_remote(local_dest_path=local_dest / name,
                                        remote_source_path=self.remote_output_dir)

class ClientTask(LocalLogger):
    _existing_names = set() # used to make sure all instances are named uniquely

    def __init__(self, app_name, app_args, client_args, positional_args, name, **kwargs):
        super().__init__(log_name=name, **kwargs)
        self.app_name = app_name
        self.app_args = app_args
        self.positional_args = positional_args
        self.client_args = client_args
        app_map = {a.name(): a for a in Application.__subclasses__()}
        if app_name not in app_map:
            raise ArgumentTypeError("Client task specified app name '{name}', but not found in registered subclasses ( {classes} )".format(
                name=app_name, classes=", ".join(app_map.keys())
            ))
        self.application = app_map[app_name]

        parser = argparse.ArgumentParser()
        ClientRuntime.add_arguments(parser=parser)
        self.application.make_client_args(parser=parser)
        self.parser=parser
        self.logging_dir = kwargs["logging_directory"]
        self._name = name
        self._results = []

    @property
    def name(self):
        return self._name

    @classmethod
    def fromJSON(cls, json_dict, **kwargs):
        app_name_key = "application"
        if app_name_key not in json_dict:
            raise ArgumentTypeError("Client json dict didn't contain required key '{k}': {d}".format(
                k=app_name_key, d=pprint.pformat(json_dict)
            ))
        kwargs["app_name"] = json_dict[app_name_key]

        app_args_key = "arguments"
        if app_args_key not in json_dict:
            raise ArgumentTypeError("Client json dict didn't contain required key '{k}': {d}".format(
                k=app_args_key, d=pprint.pformat(json_dict)
            ))
        kwargs["app_args"] = json_dict[app_args_key]

        app_pos_args_key = "positional_args"
        kwargs[app_pos_args_key] = json_dict.get(app_pos_args_key, [])

        name = json_dict["name"]

        if name in cls._existing_names:
            raise Exception("Client task name '{n}' already exists! You must name client tasks uniquely!".format(
                n=name
            ))
        else:
            cls._existing_names.add(name)
        kwargs["name"] = name

        client_args_key = "client_arguments"
        if client_args_key not in json_dict:
            raise ArgumentTypeError("Client json dosn't have required key '{k}': {d}".format(
                k=client_args_key, d=pprint.pformat(json_dict)
            ))
        kwargs["client_args"] = json_dict[client_args_key]

        return cls(**kwargs)

    def run(self, pyro_host, pyro_port, stop_event):
        def gen_args():
            for k, v in itertools.chain(self.app_args.items(), self.client_args.items()):
                yield "--{k} {v}".format(
                    k=k.replace("_", "-"), v=str(v)
                )
            yield "--pyro-ns-port {p} --pyro-ns-host {h} --master {n}".format(
                p=pyro_port, h=pyro_host, n=dist_common.pyro_master_name
            )
            for pa in self.positional_args:
                yield pa
        all_args = " ".join(gen_args())
        parsed_args = self.parser.parse_args(args=shlex.split(
            all_args
        ))

        iteration = 0
        while not stop_event.is_set():
            client = ClientRuntime()
            client.log.addHandler(
                logging.FileHandler(filename=str(self.logging_dir / "{}_{}.log".format(self.name, iteration)), mode='w')
            )
            self.log.debug("Starting {}".format(iteration))
            start_time = time.time()
            result = client.run_application(application=self.application,
                                                        args=deepcopy(parsed_args))
            end_time = time.time()
            result.update({
                "bench_start": start_time,
                "bench_end": end_time,
                "iteration": iteration
            })
            self.results.append(result)
            self.log.debug("Finished {}".format(iteration))
            iteration += 1

    @property
    def results(self):
        return self._results

class Benchmark(LocalLogger):
    _min_startup_wait = 20.0

    def __init__(self, master, workers, clients, json_dict, runtime, args, **kwargs):
        super().__init__(**kwargs)
        self.master = master
        self.runtime = runtime
        self.workers = workers
        self.clients = clients

        self.client_delay = float(json_dict.get("startup_wait", 5.0))
        if self.client_delay < 1.0:
            self.log.warning("benchmark client delay is very small ({d}). Might cause errors".format(d=self.client_delay))

        self.worker_delay = args.worker_startup_delay
        self.nice_kill_delay = args.nice_kill_delay
        self.record_finish_delay = args.record_finish_delay
        self.local_output_directory = args.output
        self.exp_description = json_dict
        self.args = args # to save for output

    def shutdown_all(self, ns):
        def shutdown_master():
            items = ns.list(prefix=dist_common.pyro_master_name)
            num_items = len(items)
            if num_items != 1:
                self.log.error("master shutdown found {num} elements registered at the prefix '{prefix}'. Picking the first one if available".format(
                    num=num_items, prefix=dist_common.pyro_master_name
                ))
            else:
                master = Pyro4.Proxy(tuple(items.values())[0])
                master.kill() # this call doesn't wait

        def shutdown_workers():
            workers = tuple(
                Pyro4.Proxy(uri) for uri in ns.list(prefix=dist_common.pyro_worker_prefix).values()
            )
            stop_and_reset_timeout = 10
            def shutdown(w, name):
                self.log.info("{} stop_and_reset...".format(name))
                try:
                    w.stop_and_reset(timeout=stop_and_reset_timeout)
                except Exception as e:
                    self.log.warning("Got exception when running stop_and_reset on {n}: {e}".format(n=name, e=e))
                else:
                    self.log.info("{} stop_and_reset complete".format(name))
            with ThreadPoolExecutor(max_workers=128) as tpe:
                tuple(tpe.map(lambda x: shutdown(w=x[0], name=x[1]), (
                    (w, ".".join((wkr.worker_name, str(wkr.worker_number))))
                    for w, wkr in zip(workers, self.workers)
                )))

        shutdown_master()
        shutdown_workers()

    # collect info on the master and workers, if applicable
    def collect_info(self):
        latencies = {
            c.name: c.results for c in self.clients
        }
        results = {
            "client_latencies": latencies
        }

        outfile = self.local_output_directory / "results.json"
        assert not outfile.exists(), "Output file already exists somehow at '{path}'".format(path=str(outfile))
        with outfile.open("w") as f:
            json.dump(fp=f, obj=results, indent=2)

        dest_path = self.local_output_directory / "config.json"
        with dest_path.open("w") as f:
            json.dump(self.exp_description, f, indent=2)

        bench_config_out = self.local_output_directory / "bench_config.json"
        with bench_config_out.open('w') as f:
            json.dump(vars(self.args), f, cls=CustomArgEncoder, indent=2)

        with ThreadPoolExecutor(max_workers=100) as tpe:
            awaiting = []
            awaiting.append(tpe.submit(self.master.copy_result_folder, local_dest=self.local_output_directory))
            awaiting.extend(
                tpe.submit(w.copy_result_folder, local_dest=self.local_output_directory)
                for w in self.workers
            )
            traces_dir = self.local_output_directory / "goodput_traces"
            assert not traces_dir.exists(), "Somehow traces dir already exists at '{}'".format(str(traces_dir))
            traces_dir.mkdir()
            awaiting.extend(
                tpe.submit(w.copy_csvs, local_directory=traces_dir)
                for w in self.workers
            )
            futures_wait(fs=awaiting)

    def run(self):
        with run_pyro_ns() as (pyro_host, pyro_port):
            with Pyro4.locateNS(host=pyro_host, port=pyro_port) as ns:
                worker_tasks = tuple(w.run(
                    pyro_host=pyro_host, pyro_port=pyro_port
                ) for w in self.workers)
                self.log.debug("Waiting for {} seconds after worker startup".format(self.worker_delay))
                time.sleep(self.worker_delay)
                self.log.debug("Done waiting for worker startup")
                master_task = self.master.run(pyro_host=pyro_host,
                                              pyro_port=pyro_port)
                try:
                    # we need the master to get all the workers started and running in order to get their output directories
                    # so min of 10 seconds is necessary
                    wait_time = max(self.client_delay, self._min_startup_wait)
                    self.log.info("Waiting {}s for system to start...".format(wait_time))
                    time.sleep(wait_time)

                    if not master_task.is_running():
                        raise Exception("Master died during startup. Got result: {}".format(master_task.wait_for_result()))
                    for wt in worker_tasks:
                        if not wt.is_running():
                            raise Exception("Worker died during startup. Got result: {}".format(wt.wait_for_result()))

                    self.log.info("Startup complete")
                    for w in self.workers:
                        w.populate_remote_output_dir(ns=ns)

                    stop_event = threading.Event()

                    thread_creator = lambda c, c_idx: threading.Thread(
                        target=c.run, kwargs = {
                            "pyro_port": pyro_port,
                            "pyro_host": pyro_host,
                            "stop_event": stop_event
                        }, name="{name}_{idx}".format(idx=c_idx,
                                                      name=c.name)
                    )

                    client_threads = tuple(
                        thread_creator(c=c, c_idx=c_idx) for c_idx, c in enumerate(self.clients)
                    )

                    for c in client_threads:
                        c.start()

                    self.log.info("Benchmark sleeping for {} seconds".format(self.runtime))
                    time.sleep(self.runtime)
                    stop_event.set()
                    self.log.info("Benchmark done sleeping")

                    for c in client_threads:
                        self.log.info("Waiting on thread: {}".format(c.name))
                        c.join()
                    self.log.info("Done waiting for all client threads to finish")

                    if not master_task.is_running():
                        raise Exception("Master died during the experiment. Got result: {}".format(master_task.wait_for_result()))
                    for wt in worker_tasks:
                        if not wt.is_running():
                            raise Exception("Worker died during the experiment. Got result: {}".format(wt.wait_for_result()))

                    self.log.debug("Waiting for {} seconds after all clients complete...".format(self.record_finish_delay))
                    time.sleep(self.record_finish_delay)

                    self.log.debug("Shutting down all clients nicely")
                    self.shutdown_all(ns=ns)

                    self.log.debug("Waiting for {} seconds after nice shutdown".format(self.nice_kill_delay))
                    time.sleep(self.nice_kill_delay)
                finally:
                    def safe_kill_item(remote_task, name):
                        if remote_task.is_running():
                            try:
                                self.log.info("Attempting to kill {}".format(name))
                                remote_task.send_signal("KILL")
                            except Exception as e:
                                self.log.warning("Got exception when attempting to kill {n}: '{e}'".format(
                                    n=name, e=e
                                ))
                            else:
                                self.log.info("Successfully killed {}".format(name))
                    with ThreadPoolExecutor(max_workers=100) as tpe:
                        # need to realize this entire result first before moving onto the next thing
                        unused = tuple(tpe.map(lambda a: safe_kill_item(*a), itertools.chain(
                            ((master_task, "master"),),
                            ((wt, "worker:{name}.{num}".format(name=w.worker_name, num=w.worker_number)) for wt, w in zip(worker_tasks, self.workers))
                        )))
                        unused = tuple(tpe.map(lambda w: w.hard_cleanup(), itertools.chain((self.master,), self.workers)))

        # nameserver is dead at this point. we can only interact through ssh commands
        self.collect_info()

def parse_experiment(file_path, username, args):
    assert isinstance(file_path, pathlib.Path)
    if not (file_path.exists() and file_path.is_file()):
        raise ArgumentTypeError("Experiment file doesn't exist at this path or isn't a file: {path}".format(
            path=str(file_path)
        ))
    with file_path.open() as f:
        exp_description = json.load(f)
    master_key = "master"
    if master_key not in exp_description:
        raise ArgumentTypeError("master key '{k}' not in the experiment description".format(k=master_key))

    time_key = "runtime"
    if time_key not in exp_description:
        raise ArgumentTypeError("runtime key '{k}' not found in experiment description [ {ks} ]".format(k=time_key,
                                                                                                        ks=", ".join(exp_description.keys())))
    runtime = exp_description[time_key]
    runtime = float(runtime)

    if runtime < min_runtime:
        raise Exception("Experiment must run for at least {} seconds".format(runtime))
    if args.runtime is not None:
        log.info("Overriding configuration file time {ct} with arg time {arg_t}".format(ct=runtime, arg_t=args.runtime))
        runtime = args.runtime

    outdir = args.output
    bench_outdir = outdir / "bench"
    assert not bench_outdir.exists(), "Somehow bench output directory '{}' already exists".format(bench_outdir)
    bench_outdir.mkdir()

    with ThreadPoolExecutor(max_workers=32) as tpe:
        try:
            master = tpe.submit(fn=Master.fromJSON, json_dict=exp_description[master_key], username=username, output_directory=outdir,
                                     logging_directory=bench_outdir, log_name="master")
        except Exception as e:
            raise ArgumentTypeError("Trying to parse master gave the following exception: '{e}'".format(e=e))

        worker_key = "workers"
        if worker_key not in exp_description:
            raise ArgumentTypeError("worker key '{k}' not in the experiment description".format(k=worker_key))
        worker_descriptions = exp_description[worker_key]
        if len(worker_descriptions) == 0:
            raise ArgumentTypeError("No workers provided")

        make_worker = partial(Worker.fromJSON,
                              username=username,
                              output_directory=outdir,
                              startup_delay=args.worker_startup_delay,
                              startup_sleep_interval=args.worker_startup_sleep_interval,
                              logging_directory=bench_outdir,
                              sleep_interval=args.worker_sleep_interval,)

        workers = tpe.map(lambda a: make_worker(json_dict=a), worker_descriptions)

        client_task_key = "clients"
        if client_task_key not in exp_description:
            raise ArgumentTypeError("client key '{k}' not in the experiment description".format(k=client_task_key))
        client_task_descriptions = exp_description[client_task_key]
        if len(client_task_descriptions) == 0:
            raise ArgumentTypeError("Expected >0 client tasks")

        def make_clients():
            for ctd in client_task_descriptions:
                assert isinstance(ctd, dict)
                yield ClientTask.fromJSON(json_dict=ctd, logging_directory=bench_outdir)

        client_tasks = tuple(make_clients())

        return Benchmark(
            workers=tuple(workers),
            master=master.result(),
            clients=client_tasks,
            json_dict=exp_description,
            args=args,
            runtime=runtime,
            logging_directory=bench_outdir,
            log_name="bench"
        )

def add_args(parser):
    parser.add_argument("-u", "--username", default="whitlock", help="username to ssh into all machines. Have ssh keys to access the machines via this username!!")
    parser.add_argument("--worker-startup-delay", default=8, type=parse.numeric_min_checker(minimum=0.1, numeric_type=float, message="need a minimal startup delay of 0.1"),
                        help="seconds to wait from starting up workers until starting the master")
    parser.add_argument("--worker-sleep-interval", default=5, type=parse.numeric_min_checker(minimum=1.0, numeric_type=float, message="minimum of 1 sec startup delay"), help="delay when startup up a worker")
    parser.add_argument("--worker-startup-sleep-interval", default=2, type=parse.numeric_min_checker(minimum=0.5, numeric_type=float, message="0.5s min for startup interval"), help="startup interval for ")
    parser.add_argument("--record-finish-delay", default=5, type=parse.numeric_min_checker(minimum=0.1, numeric_type=float, message="minimum nice kill delay 0.1"),
                        help="delay after nicely killing cluster after experiment")
    parser.add_argument("--experiment-time", dest="runtime", type=parse.numeric_min_checker(minimum=min_runtime, numeric_type=float, message="minimum runtime for the experiment"),
                        help="time to run the experiment for. Overrides what is specified in the file.")
    parser.add_argument("--nice-kill-delay", default=5, type=parse.numeric_min_checker(minimum=0.1, numeric_type=float, message="minimum nice kill delay 0.1"),
                        help="delay after nicely killing cluster after experiment")
    parser.add_argument("-o", "--output", default="results", type=lambda p: pathlib.Path(p).absolute(), help="path to output the results to")
    parser.add_argument("experiment", type=pathlib.Path, help="path to a JSON file describing the experiment setup")

def run(args):
    # need to do this so the ClientTasks can search registered subclasses# need to do this so the ClientTasks can search registered subclasses
    benchmark_name = "bench_run"
    log = logging.getLogger(name=benchmark_name)

    output_dir = args.output
    if output_dir.exists():
        dir_name = output_dir.name
        countup = itertools.count()
        parent = output_dir.parent
        while output_dir.exists():
            output_dir = parent / "{name}_{num}".format(name=dir_name, num=next(countup))
        log.warning("Original directory '{orig}' exists already! Using output directory '{od}'".format(
            orig=str(args.output), od=str(output_dir)
        ))
    output_dir.mkdir(parents=True)
    args.output = output_dir

    util.import_submodules(app)
    experiment = parse_experiment(file_path=args.experiment,
                                  username=args.username, args=args)
    experiment.run()