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
from app.app import Application
import git
from . import dist_common
import logging
import json
import pathlib
import inspect
logging.basicConfig(level=logging.DEBUG)

class CustomArgEncoder(json.JSONEncoder):
    def default(self, o):
        if inspect.isclass(o):
            if issubclass(o, Runtime):
                return o.name()
            elif issubclass(o, Application):
                return o.name()
        else:
            if isinstance(o, pathlib.PurePath):
                return str(o)
        return super().default(o)

class Runtime(abc.ABC):
    def __init__(self):
        self.log = logging.getLogger(name=self.name())
        self.log.setLevel(level=logging.DEBUG)

    @staticmethod
    @abc.abstractmethod
    def name():
        """
        Return a string to be used for the top-level call
        :return:
        """
        pass

    @staticmethod
    def add_record_args(parser):
        parser.add_argument("--record-args", default=False, action="store_true", help="record all arguments passed to this script invocation")
        dist_common.add_common_record_args(parser=parser)

    def write_out_args(self, args, encoder=CustomArgEncoder):
        assert issubclass(encoder, json.JSONEncoder), "argument encoder must subclass json.JSONEncoder"
        arg_vars = vars(args)
        shell_repo = git.Repo(str((pathlib.Path(__file__).parent)), search_parent_directories=True)
        arg_vars["shell_git"] = shell_repo.head.object.hexsha
        possible_barrier_path = pathlib.Path(shell_repo.working_dir).absolute().parent / "barrier"

        try:
            arg_vars["gate_git"] = git.Repo(str(possible_barrier_path)).head.object.hexsha
        except git.InvalidGitRepositoryError:
            self.log.warning("Unable to attach barrier repo hexsha at path '{}'".format(str(possible_barrier_path)))

        out_dir = args.output_directory
        assert out_dir.exists()
        args_file = out_dir / "args.json"
        with args_file.open(mode="w") as f:
            json.dump(arg_vars, f, cls=encoder, indent=2)

    @staticmethod
    def help_message():
        return ""

    @classmethod
    @abc.abstractmethod
    def add_arguments(cls, parser):
        pass

    def run_application(self, application, args):
        assert issubclass(application, Application)
        return self._run_application(ApplicationClass=application, args=args)

    @abc.abstractmethod
    def _run_application(self, ApplicationClass, args):
        pass

    @classmethod
    def populate_app_args(cls, parser, app):
        assert issubclass(app, Application)
        cls._populate_app_args(parser=parser, app=app)

    @staticmethod
    @abc.abstractmethod
    def _populate_app_args(parser, app):
        pass
