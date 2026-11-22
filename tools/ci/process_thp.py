"""Apply an explicitly requested, process-local THP control before exec."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=("inherit", "disable"), required=True)
parser.add_argument("--record", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or not Path(command[0]).is_absolute():
    parser.error("An absolute executable after -- is required")
if args.record.exists():
    parser.error("The transition record must not already exist")

libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
libc.prctl.restype = ctypes.c_int
previous = libc.prctl(42, 0, 0, 0, 0)
if previous not in (0, 1):
    raise OSError(ctypes.get_errno(), "PR_GET_THP_DISABLE failed")
if args.policy == "disable" and libc.prctl(41, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
effective = libc.prctl(42, 0, 0, 0, 0)
assert effective == (1 if args.policy == "disable" else previous)

# exec preserves this flag on Linux. Recording before exec keeps the exact
# numerical program unmodified while making the changed runtime policy visible.
record = {
    "policy": args.policy,
    "pid": os.getpid(),
    "previous_disable_flag": previous,
    "effective_disable_flag": effective,
    "command": command,
    "launcher": str(Path(__file__).resolve()),
    "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "shared_policy_changed": False,
}
args.record.parent.mkdir(parents=True, exist_ok=True)
with args.record.open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.execv(command[0], command)
