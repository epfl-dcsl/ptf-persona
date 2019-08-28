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
import json
import pathlib
import shutil
from argparse import ArgumentTypeError

import logging
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

def yes_or_no(question):
    # could this overflow the stack if the user was very persistent?
    reply = str(input(question+' (y/n): ')).lower().strip()
    valid_yes = { 'y', 'yes'}
    valid_no = { 'n', 'no' }
    valid_all = valid_yes.union(valid_no)
    while len(reply) == 0 or reply not in valid_all:
        reply = str(input(question+' (y/n): ')).lower().strip()
    return reply in valid_yes

def numeric_min_checker(minimum, message, numeric_type=int):
    def check_number(n):
        n = numeric_type(n)
        if n < minimum:
            raise ArgumentTypeError("{msg}: got {got}, minimum is {minimum}".format(
                msg=message, got=n, minimum=minimum
            ))
        return n
    return check_number

def path_exists_checker(check_dir=True, make_absolute=True, make_if_empty=False, rm_if_exists=False):
    def _func(path):
        path = pathlib.Path(path)
        if make_absolute:
            path = path.absolute()
        if path.exists():
            if check_dir:
                if not path.is_dir():
                    raise ArgumentTypeError("path {pth} exists, but isn't a directory".format(pth=path))
            elif not path.is_file():
                raise ArgumentTypeError("path {pth} exists, but isn't a file".format(pth=path))

            if rm_if_exists:
                if path.is_dir():
                    shutil.rmtree(path=str(path))
                    if not make_if_empty:
                        log.warning("rm_if_exists=True, but make_if_empty=False for a directory. Making it anyway (path: {p})".format(
                            p=str(path)
                        ))
                    path.mkdir(parents=True)
                else:
                    assert path.is_file()
                    path.unlink()
        elif check_dir and make_if_empty:
            path.mkdir(parents=True)
        else:
            raise ArgumentTypeError("path {pth} doesn't exist on filesystem".format(pth=path))

        return path
    return _func

def non_empty_string_checker(string):
    if len(string) == 0:
        raise ArgumentTypeError("string is empty!")
    return string

filepath_key = "filepath"
def add_dataset(parser):
    """
    Adds the dataset, including parsing, to any parser / subparser
    """
    def dataset_parser(filename):
        path = pathlib.Path(filename)
        if not path.exists():
            raise ArgumentTypeError("AGD metadata file not present at {}".format(filename))
        with path.open("r") as f:
            try:
                loaded = json.load(f)
                loaded[filepath_key] = path
                return loaded
            except json.JSONDecodeError:
                log.error("Unable to parse AGD metadata file {}".format(filename))
                raise

    parser.add_argument("dataset", type=dataset_parser, help="The AGD json metadata file describing the dataset")
