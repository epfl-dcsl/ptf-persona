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
import subprocess
import shutil
import pathlib
import threading
import os
import shlex
import csv
import signal
from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor
import socket
import psutil
from .dist_common import get_external_ip_addr
import time

import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)
log.setLevel(level=logging.DEBUG)

def check_command(cmd):
    if shutil.which(cmd) is None:
        raise Exception("Need to install package for command '{c}'. Probably sysstat".format(c=cmd))

def get_cluster_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("iccluster001", 22))
    return s.getsockname()[0]

def get_cluster_iface():
    cluster_ip = get_cluster_ip()
    for iface_name, addresses in psutil.net_if_addrs().items():
        for address in addresses:
            if address.address == cluster_ip:
                return iface_name
    else:
        raise Exception("Couldn't find cluster interface with ip {}".format(cluster_ip))

@contextmanager
def StatsRecorder(pid_to_outfile_map, kill_self=False):
    """
    :param pid_to_outfile_map: a mapping of {pid: pathlib.Path()}, where the value is the file to write the given pid info to
    :return:
    """
    check_command("pidstat")

    extra_option = ""
    pidstat_options = subprocess.run(["pidstat", "-H"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if pidstat_options.returncode == 0:
        extra_option = "HI"

    pid_processes = []
    try:
        for pid, outfile in pid_to_outfile_map.items():
            assert isinstance(pid, int)
            assert isinstance(outfile, pathlib.Path)
            # We use the "build a list" style instead of generating the Popen instances so we can kill them if needed to bail early
            pid_processes.append((pid, subprocess.Popen("pidstat -hrduvwR{extra} -p {pid} 1 | sed '1d;/^[#]/{{4,$d}};/^[#]/s/^[#][ ]*//;/^$/d;s/^[ ]*//;s/[ ]\+/,/g' > {outfile}".format(
                pid=pid, outfile=str(outfile.absolute()), extra=extra_option
            ), shell=True, universal_newlines=True, stdout=subprocess.PIPE,
                start_new_session=True))) # needed so we can kill by pgid

        yield
    finally:
        def kill_pid(monitored_pid, proc):
            if monitored_pid != os.getpid() or kill_self:
                print("Killing non-self pid: {}".format(monitored_pid))
                # if we are monitoring ourself, we don't need to kill our own process unless we have to
                # kill the process will create a partial line (most likely), so if possible it's better to
                # allow a clean shutdown, where pidstat just stops on its own
                pgid = os.getpgid(proc.pid)
                try:
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                        if proc.poll() is None:
                            break
                        else:
                            os.killpg(pgid, sig)
                            time.sleep(1.0)
                except ProcessLookupError:
                    log.warning("Can't kill pid {p}. Not found".format(p=proc.pid))
        with ThreadPoolExecutor(max_workers=8) as tpe:
            tpe.map(kill_pid, pid_processes)

@contextmanager
def NetRecorder(outdir, record_interval=1.0):
    outpath = outdir / "net_stat.csv"

    ext_ip = get_external_ip_addr()
    iface_name = None
    for name, iface_addrs in psutil.net_if_addrs().items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET and addr.address == ext_ip:
                iface_name = name
    if iface_name is None:
        raise Exception("Couldn't find interface for IPv4 address: '{ip}'".format(ip=ext_ip))

    stop_event = threading.Event()

    net_stats = []
    def collect_net_stats():
        prior_counters = None
        prior_time = None
        while not stop_event.is_set():
            current_time = time.time()
            current_counters = psutil.net_io_counters(pernic=True)
            if prior_counters is not None:
                prior_nic = prior_counters[iface_name]
                current_nic = current_counters[iface_name]
                elapsed = current_time - prior_time
                assert elapsed > 0, "Time diff was not >0. Got {}".format(elapsed)
                elapsed = float(elapsed)
                rx_bytes = current_nic.bytes_recv - prior_nic.bytes_recv
                assert rx_bytes >= 0, "Got a negative rx byte diff {}".format(rx_bytes)
                tx_bytes = current_nic.bytes_sent - prior_nic.bytes_sent
                assert tx_bytes >= 0, "Got a negative tx byte diff {}".format(tx_bytes)
                rx_packets = current_nic.packets_recv - prior_nic.packets_recv
                assert rx_packets >= 0, "Got a negative rx byte diff {}".format(rx_packets)
                tx_packets = current_nic.packets_sent - prior_nic.packets_sent
                assert tx_packets >= 0, "Got a negative tx byte diff {}".format(tx_packets)
                net_stats.append({
                    "time": current_time,
                    "rx_bytes/sec": rx_bytes / elapsed,
                    "tx_bytes/sec": tx_bytes / elapsed,
                    "rx_packets/sec": rx_packets / elapsed,
                    "tx_packets/sec": tx_packets / elapsed,
                    "rx_bytes": current_nic.bytes_recv,
                    "tx_bytes": current_nic.bytes_sent,
                    "rx_packets": current_nic.packets_recv,
                    "tx_packets": current_nic.packets_sent
                })
            prior_counters = current_counters
            prior_time = current_time
            time.sleep(record_interval)

    start_thread = threading.Thread(target=collect_net_stats)

    try:
        start_thread.start()
        log.info("Starting net recorder.")
        yield
    finally:
        log.info("Stopping net recorder")
        stop_event.set()
        if start_thread.is_alive():
            start_thread.join()
            if len(net_stats) > 0:
                with outpath.open("w") as f:
                    writer = csv.DictWriter(f=f, fieldnames=net_stats[0].keys())
                    writer.writeheader()
                    writer.writerows(net_stats)
            else:
                log.warning("Net stats didn't record anything")
        else:
            log.error("Net recorder didn't start correctly")
        log.debug("Net recorder stopped")

@contextmanager
def record_self(outdir, kill_self=False, net_record_interval=1.0):
    assert isinstance(outdir, pathlib.Path), "record_self(outdir) must be passed a pathlib.Path parameter"

    if not outdir.exists():
        raise Exception("outdir '{d}' doesn't exist. Caller must make it!".format(d=str(outdir)))

    outfile = outdir / "stats.csv"
    if outfile.exists():
        outfile.unlink()

    with ExitStack() as es:
        es.enter_context(StatsRecorder(
            pid_to_outfile_map={os.getpid(): outfile},
            kill_self=kill_self
        ))
        es.enter_context(NetRecorder(outdir=outdir, record_interval=net_record_interval))
        yield

def recording_cleanup():
    subprocess.run(shlex.split("pkill sar"), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    subprocess.run(shlex.split("pkill sadc"), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
