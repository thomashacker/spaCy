"""
Permits the execution of a parallel command group.

The main process starts a worker process for each spaCy projects command in the parallel group.
Each spaCy projects command consists of n OS level commands. The worker process starts
subprocesses for these OS level commands serially, and sends status information about their
execution back to the main process.

The main process maintains a state machine for each spaCy projects command/worker process. The meaning
of the states is documented alongside the _ParallelCommandInfo.STATES code. Note that the distinction
between the states 'failed' and 'terminated' is not meaningful on Windows, so that both are displayed
as 'failed/terminated' on Windows systems.
"""
from typing import Any, List, Optional, Dict, cast, Iterator, ClassVar, Tuple
import os
import sys
import queue
from pathlib import Path
from time import time
from multiprocessing import Manager, get_context
from multiprocessing.context import SpawnProcess
from tempfile import mkdtemp
from signal import SIGTERM
from subprocess import STDOUT, Popen, TimeoutExpired
from dataclasses import dataclass, field
from wasabi import msg
from wasabi.util import color, supports_ansi

from .._util import check_rerun, check_deps, update_lockfile, load_project_config
from ...util import SimpleFrozenDict, working_dir, split_command, join_command
from ...util import check_bool_env_var, ENV_VARS
from ...errors import Errors

# Use spawn to create worker processes on all OSs for consistency
_mp_context = get_context("spawn")

# How often the worker processes managing the commands in a parallel group
# send keepalive messages to the main process (seconds)
_PARALLEL_GROUP_STATUS_INTERVAL = 1

# The maximum permissible width of divider text describing a parallel command group
_MAX_WIDTH_DIVIDER = 60

# Whether or not to display realtime status information
_DISPLAY_STATUS_TABLE = sys.stdout.isatty() and supports_ansi()


@dataclass
class _Message:
    type_: ClassVar[str]
    cmd_index: int


@dataclass
class _Started(_Message):
    type_ = "started"
    os_cmd_index: int
    pid: int


@dataclass
class _KeepAlive(_Message):
    type_ = "keepalive"


@dataclass
class _Completed(_Message):
    type_ = "completed"
    rc: int
    console_output: str


class _ParallelCommand:
    @dataclass
    class _ParallelCommandState:
        name: str
        display_color: str
        # transitions: the names of states that can legally follow this state
        transitions: List = field(default_factory=list)

    STATES = (
        # The command has not yet been run
        _ParallelCommandState(
            "pending", "yellow", ["starting", "cancelled", "not rerun"]
        ),
        # The command was not run because its settings, inputs and outputs had not changed
        # since a previous successful run
        _ParallelCommandState("not rerun", "blue"),
        # The main process has spawned a worker process for the command but not yet received
        # an acknowledgement
        _ParallelCommandState("starting", "green", ["running", "failed", "hung"]),
        # The command is running
        _ParallelCommandState(
            "running", "green", ["running", "succeeded", "failed", "terminated", "hung"]
        ),
        # The command succeeded (rc=0)
        _ParallelCommandState("succeeded", "green"),
        # The command failed (rc>0)
        _ParallelCommandState("failed", "red"),
        # The command was terminated (rc<0), usually but not necessarily by the main process because
        # another command within the same parallel group failed
        _ParallelCommandState("terminated", "red"),
        # The main process would expect the worker process to be running, but the worker process has
        # not communicated with the main process for an extended period of time
        _ParallelCommandState("hung", "red"),
        # The main process did not attempt to execute the command because another command in the same
        # parallel group had already failed
        _ParallelCommandState("cancelled", "red"),
    )

    state_dict = {state.name: state for state in STATES}

    def __init__(
        self, name: str, index: int, config_command: Dict, dry: bool, temp_log_dir: Path
    ):
        self.name = name
        self.index = index
        self.config_command = config_command
        self.os_cmds = config_command["script"]
        self.dry = dry
        self.log_path = temp_log_dir / (self.name + ".log")
        # we can use the command name as a unique log filename because a parallel
        # group is not allowed to contain the same command more than once

        self.state = _ParallelCommand.state_dict["pending"]
        self.worker_process: Optional[SpawnProcess] = None
        self.pid: Optional[int] = None
        self.last_keepalive_time: Optional[int] = None
        self.running_os_cmd_index: Optional[int] = None

    def start(self, parallel_group_status_queue) -> None:
        self.change_state("starting")
        self.last_keepalive_time = int(time())
        self.worker_process = _mp_context.Process(
            target=_project_run_parallel_cmd,
            args=[],
            kwargs={
                "cmd_index": self.index,
                "os_cmds": self.os_cmds,
                "dry": self.dry,
                "log_path": self.log_path,
                "parallel_group_status_queue": parallel_group_status_queue,
            },
        )
        self.worker_process.start()

    def terminate_if_alive(self):
        if self.worker_process is not None and self.worker_process.is_alive():
            self.worker_process.terminate()

    def change_state(self, new_state: str):
        if new_state not in self.state.transitions:
            raise RuntimeError(
                Errors.E1044.format(old_state=self.state.name, new_state=new_state)
            )
        self.state = _ParallelCommand.state_dict[new_state]

    @property
    def state_repr(self) -> str:
        state_str = self.state.name
        if state_str == "running" and self.running_os_cmd_index is not None:
            state_str = (
                f"{state_str} ({self.running_os_cmd_index + 1}/{len(self.os_cmds)})"
            )
        elif state_str in ("failed", "terminated") and os.name == "nt":
            state_str = "failed/terminated"
        # we know ANSI commands are available because otherwise
        # the status table would not be being displayed in the first place
        return color(state_str, self.state.display_color)


def project_run_parallel_group(
    project_dir: Path,
    cmd_names: List[str],
    *,
    overrides: Dict[str, Any] = SimpleFrozenDict(),
    force: bool = False,
    dry: bool = False,
) -> None:
    """Run a parallel group of commands. Note that because of the challenges of managing
    parallel console output streams it is not possible to specify a value for 'capture' as
    when executing commands serially. Essentially, with parallel groups 'capture==True'.

    project_dir (Path): Path to project directory.
    cmd_names: the names of the spaCy projects commands within the parallel group.
    overrides (Dict[str, Any]): Optional config overrides.
    force (bool): Force re-running, even if nothing changed.
    dry (bool): Perform a dry run and don't execute commands.
    """
    temp_log_dir = Path(mkdtemp())
    _print_divider(cmd_names, temp_log_dir)

    config = load_project_config(project_dir, overrides=overrides)
    parallel_cmds = _read_commands_from_config(
        config, cmd_names, project_dir, dry, temp_log_dir
    )
    max_parallel_processes = config.get("max_parallel_processes", len(parallel_cmds))
    parallel_group_status_queue = Manager().Queue()

    with working_dir(project_dir) as current_dir:
        for parallel_cmd in parallel_cmds:
            if not _is_cmd_to_rerun(parallel_cmd, project_dir, current_dir, dry, force):
                parallel_cmd.change_state("not rerun")
                continue
            
        next_cmd_to_start_iter = iter(c for c in parallel_cmds if c.state.name != "not rerun")
        for _ in range(max_parallel_processes):
            next_cmd = next(next_cmd_to_start_iter, None)
            if next_cmd is not None:
                next_cmd.start(parallel_group_status_queue)
            else:
                break

        completed_messages = _process_worker_status_messages(
            parallel_cmds,
            next_cmd_to_start_iter,
            parallel_group_status_queue,
            current_dir,
            dry,
        )
        _report_command_states(
            [(c.name, c.state.name, completed_messages[i])
            for i, c in enumerate(parallel_cmds)
            ])
        # Occasionally when something has hung there may still be worker processes to tidy up
        for cmd in parallel_cmds:
            cmd.terminate_if_alive()
        group_rc = _get_group_rc(completed_messages)
        if group_rc > 0:
            msg.fail("A command in the parallel group failed.")
            sys.exit(group_rc)
        elif group_rc < 0:
            msg.fail("Command(s) in the parallel group hung or were terminated.")
            sys.exit(group_rc)


def _process_worker_status_messages(
    parallel_cmds: List[_ParallelCommand],
    next_cmd_to_start_iter: Iterator[_ParallelCommand],
    parallel_group_status_queue: queue.Queue,
    current_dir: Path,
    dry: bool,
) -> List[Optional[_Completed]]:
    """Listen on the status queue and process messages received from the worker processes.

    cmd_infos: a list of info objects about the commands in the parallel group.
    proc_to_cmd_infos: a dictionary from Process objects to command info objects.
    parallel_group_status_queue: the status queue.
    worker_process_iterator: an iterator over the processes, some or all of which
        will already have been iterated over and started.
    current_dir: the current directory.
    dry (bool): Perform a dry run and don't execute commands.
    """
    status_table_not_yet_displayed = True
    completed_messages: List[Optional[_Completed]] = [None for _ in parallel_cmds]
    while any(
        cmd.state.name in ("pending", "starting", "running") for cmd in parallel_cmds
    ):
        try:
            mess = parallel_group_status_queue.get(
                timeout=_PARALLEL_GROUP_STATUS_INTERVAL * 20
            )
        except queue.Empty:
            # No more messages are being received: the whole group has hung
            _cancel_hung_group(parallel_cmds)
            break
        _check_for_hung_cmds(parallel_cmds)
        cmd = parallel_cmds[mess.cmd_index]
        if isinstance(mess, (_Started, _KeepAlive)):
            cmd.last_keepalive_time = int(time())
        if isinstance(mess, _Started):
            cmd.change_state("running")
            cmd.running_os_cmd_index = mess.os_cmd_index
            cmd.pid = mess.pid
        elif isinstance(mess, _Completed):
            completed_messages[mess.cmd_index] = mess
            if mess.rc == 0:
                cmd.change_state("succeeded")
                if not dry:
                    update_lockfile(current_dir, cmd.config_command)
                next_cmd = next(next_cmd_to_start_iter, None)
                if next_cmd is not None:
                    next_cmd.start(parallel_group_status_queue)
            elif mess.rc > 0:
                cmd.change_state("failed")
            else:
                cmd.change_state("terminated")
        if any(
            c for c in parallel_cmds if c.state.name in ("failed", "terminated", "hung")
        ):
            # a command in the group hasn't succeeded, so terminate/cancel the rest
            _cancel_failed_group(parallel_cmds)
        if not isinstance(mess, _KeepAlive) and _DISPLAY_STATUS_TABLE:
            if status_table_not_yet_displayed:
                status_table_not_yet_displayed = False
            else:
                # overwrite the existing status table
                print("\033[2K\033[F" * (4 + len(parallel_cmds)))
            data = [[c.name, c.state_repr] for c in parallel_cmds]
            header = ["Command", "Status"]
            msg.table(data, header=header)
    return completed_messages


def _cancel_failed_group(parallel_cmds):
    for cmd in (c for c in parallel_cmds if c.state.name == "running"):
        try:
            os.kill(cast(int, cmd.pid), SIGTERM)
        except ProcessLookupError:
            # the subprocess the main process is trying to kill could already
            # have completed, and the message from the worker process notifying
            # the main process about this could still be in the queue
            pass
    for cmd in (c for c in parallel_cmds if c.state.name == "pending"):
        cmd.change_state("cancelled")


def _cancel_hung_group(parallel_cmds):
    for cmd in (c for c in parallel_cmds if c.state.name in ("starting", "running")):
        cmd.change_state("hung")
    for cmd in (c for c in parallel_cmds if c.state.name == "pending"):
        cmd.change_state("cancelled")


def _check_for_hung_cmds(parallel_cmds):
    for cmd in (c for c in parallel_cmds if c.state.name in ("starting", "running")):
        if (
            cmd.last_keepalive_time is not None
            and time() - cmd.last_keepalive_time > _PARALLEL_GROUP_STATUS_INTERVAL * 20
        ):
            # a specific command has hung
            cmd.change_state("hung")


def _read_commands_from_config(
    config, cmd_names: List[str], project_dir, dry, temp_log_dir
) -> List[_ParallelCommand]:
    config_cmds = {cmd["name"]: cmd for cmd in config.get("commands", [])}
    for cmd_name in config_cmds:
        check_deps(config_cmds[cmd_name], cmd_name, project_dir, dry)
    return [
        _ParallelCommand(name, index, config_cmds[name], dry, temp_log_dir)
        for index, name in enumerate(cmd_names)
    ]


def _is_cmd_to_rerun(parallel_cmd, project_dir, current_dir, dry, force) -> bool:
    check_spacy_commit = check_bool_env_var(ENV_VARS.PROJECT_USE_GIT_VERSION)
    return (
        check_rerun(
            current_dir,
            parallel_cmd.config_command,
            check_spacy_commit=check_spacy_commit,
        )
        or force
    )


def _report_command_states(cmd_states: List[Tuple[str, str, Optional[_Completed]]]):
    for name, state, completed_message in cmd_states:
        if state != "cancelled":
            msg.divider(name)
            if state == "not rerun":
                msg.info(f"Skipping '{name}': nothing changed")
            elif completed_message is not None:
                print(completed_message.console_output)


def _print_divider(cmd_names: List[str], temp_log_dir: Path) -> None:
    divider_parallel_descriptor = parallel_descriptor = (
        "parallel[" + ", ".join(cmd_names) + "]"
    )
    if len(divider_parallel_descriptor) > _MAX_WIDTH_DIVIDER:
        divider_parallel_descriptor = (
            divider_parallel_descriptor[: (_MAX_WIDTH_DIVIDER - 3)] + "..."
        )
    msg.divider(divider_parallel_descriptor)
    if not _DISPLAY_STATUS_TABLE and len(parallel_descriptor) > _MAX_WIDTH_DIVIDER:
        # reprint the descriptor if it was too long and had to be cut short
        print(parallel_descriptor)
    msg.info("Temporary logs are being written to " + str(temp_log_dir))


def _get_group_rc(completed_messages: List[Optional[_Completed]]) -> int:
    set_rcs = [mess.rc for mess in completed_messages if mess is not None]
    if len(set_rcs) == 0:
        return 0
    max_rc = max(set_rcs)
    if max_rc > 0:
        return max_rc
    min_rc = min(set_rcs)
    if min_rc < 0:
        return min_rc
    return 0


def _project_run_parallel_cmd(
    *,
    cmd_index: int,
    os_cmds: Dict,
    dry: bool,
    log_path: Path,
    parallel_group_status_queue: queue.Queue,
) -> None:
    """Run a single spaCy projects command as a worker process.

    Communicates with the main process via queue messages whose type is determined
    by the entry 'mess_type' and that are structured as dictionaries. Possible
    values of 'mess_type' are 'started', 'completed' and 'keepalive'. Each dictionary
    type contains different additional fields.

    dry (bool): Perform a dry run and don't execute commands.
    log_path: the temporary file path to which to log.
    parallel_group_status_queue: the queue via which to send status messages to the
        main process.
    """
    # buffering=0: make sure output is not lost if a subprocess is terminated
    # The return code will be 0 if the group succeeded,
    # or the return code of the first failing item.
    with log_path.open("wb", buffering=0) as logfile:
        return_code = _run_os_cmds(
            cmd_index,
            os_cmds,
            logfile,
            parallel_group_status_queue,
            dry,
        )

    with log_path.open("r") as logfile:
        parallel_group_status_queue.put(
            _Completed(
                cmd_index=cmd_index,
                rc=return_code,
                console_output=logfile.read(),
            )
        )


def _run_os_cmds(
    cmd_index: int, os_cmds: Dict, logfile, status_queue: queue.Queue, dry: bool
) -> int:
    for os_cmd_index, os_cmd in enumerate(os_cmds):
        command = _get_os_cmd(os_cmd)
        logfile.write(
            bytes(
                f"Running command: {join_command(command)}" + os.linesep,
                encoding="utf8",
            )
        )
        if dry:
            status_queue.put(
                _Started(
                    cmd_index=cmd_index,
                    os_cmd_index=os_cmd_index,
                    pid=-1,
                )
            )
        else:
            rc = _run_os_cmd(
                cmd_index,
                os_cmd_index,
                command,
                status_queue,
                logfile,
            )
            if rc != 0:
                return rc
    return 0


def _get_os_cmd(os_cmd: str) -> List[str]:
    command = split_command(os_cmd)
    if len(command) and command[0] in ("python", "python3"):
        # -u: prevent buffering within Python
        command = [sys.executable, "-u", *command[1:]]
    elif len(command) and command[0].startswith("python3"):  # e.g. python3.10
        command = [command[0], "-u", *command[1:]]
    elif len(command) and command[0] in ("pip", "pip3"):
        command = [sys.executable, "-m", "pip", *command[1:]]
    return command


def _run_os_cmd(
    cmd_index: int,
    os_cmd_index: int,
    command: List[str],
    queue: queue.Queue,
    logfile,
) -> int:
    try:
        sp = Popen(
            command,
            stdout=logfile,
            stderr=STDOUT,
            env=os.environ.copy(),
            encoding="utf8",
        )
    except FileNotFoundError:
        # Indicates the *command* wasn't found, it's an error before the command
        # is run.
        logfile.write(
            bytes(
                Errors.E970.format(
                    str_command=" ".join(command),
                    tool=command[0],
                ),
                encoding="UTF-8",
            )
        )
        return 1

    queue.put(
        _Started(
            cmd_index=cmd_index,
            os_cmd_index=os_cmd_index,
            pid=sp.pid,
        )
    )

    while sp.returncode == None:
        try:
            sp.wait(timeout=_PARALLEL_GROUP_STATUS_INTERVAL)
        except TimeoutExpired:
            pass
        if sp.returncode == None:
            queue.put(
                _KeepAlive(
                    cmd_index=cmd_index,
                )
            )
        elif sp.returncode != 0:
            if os.name == "nt":
                status = "Failed/terminated"
            elif sp.returncode > 0:
                status = "Failed"
            else:
                status = "Terminated"
            logfile.write(
                bytes(
                    os.linesep + f"{status} (rc={sp.returncode})" + os.linesep,
                    encoding="UTF-8",
                )
            )
    return sp.returncode
