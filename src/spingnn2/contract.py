"""Portable v1 wire format: UTF-8 JSON and aligned, little-endian C-order arrays."""

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

FORMAT = "spingnn2.contract.v1"
VERSION = 1
FAMILIES = (
    "Allegro_nosoc",
    "Allegro_soc",
    "Allegro_born",
    "Allegro_e2IP",
    "Allegro_soc_e2IP",
)
_DTYPES = {"<f8", "<f4", "<i8", "<i4", "<i2", "|i1", "|u1", "|b1"}


@contextmanager
def _directory_lock(directory, exclusive):
    # Both conventional filenames must switch together for concurrent readers.
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)


def write_contract(stem, model):
    stem = str(stem).removesuffix(".sp2.json")
    manifest, binary = Path(stem + ".sp2.json"), Path(stem + ".sp2.bin")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    digest = hashlib.sha256()
    temporary = []
    try:
        with NamedTemporaryFile(dir=manifest.parent, delete=False) as blob:
            temporary.append(Path(blob.name))

            def encode(value):
                if isinstance(value, np.ndarray):
                    dtype = value.dtype.newbyteorder("<")
                    if dtype.str not in _DTYPES:
                        raise ValueError(f"Unsupported contract dtype {dtype}")
                    padding = bytes((-blob.tell()) % 8)
                    blob.write(padding)
                    digest.update(padding)
                    offset = blob.tell()
                    raw = value.astype(dtype, copy=False).tobytes(order="C")
                    blob.write(raw)
                    digest.update(raw)
                    arrays.append(
                        dict(offset=offset, shape=list(value.shape), dtype=dtype.str)
                    )
                    return {"$array": len(arrays) - 1}
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, dict):
                    return {k: encode(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [encode(v) for v in value]
                return value

            model_json = encode(model)
        header = dict(
            format=FORMAT,
            format_version=VERSION,
            binary=binary.name,
            sha256=digest.hexdigest(),
            arrays=arrays,
            model=model_json,
        )
        with NamedTemporaryFile(mode="w", dir=manifest.parent, delete=False) as out:
            temporary.append(Path(out.name))
            json.dump(header, out, indent=2, allow_nan=False)
            out.write("\n")
        with _directory_lock(manifest.parent, True):
            temporary[0].replace(binary)
            temporary[1].replace(manifest)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return manifest, binary


def load_contract(manifest):
    return load_contract_with_digest(manifest)[0]


def load_contract_with_digest(manifest):
    """Decode a contract and hash the exact manifest under the same read lock."""
    manifest = Path(manifest)
    with _directory_lock(manifest.parent, False):
        encoded = manifest.read_bytes()
        header = json.loads(encoded)
        if header.get("format") != FORMAT or header.get("format_version") != VERSION:
            raise ValueError("Unsupported SpinGNN2 contract format or version")
        binary = header.get("binary")
        if not isinstance(binary, str) or Path(binary).name != binary:
            raise ValueError("Invalid contract binary filename")
        blob = manifest.with_name(binary).read_bytes()
    if hashlib.sha256(blob).hexdigest() != header.get("sha256"):
        raise ValueError("SpinGNN2 binary SHA-256 mismatch")
    arrays = []
    for desc in header["arrays"]:
        shape, offset, dtype = desc["shape"], desc["offset"], desc["dtype"]
        if (
            dtype not in _DTYPES
            or type(offset) is not int
            or offset < 0
            or offset % 8
            or any(type(n) is not int or n < 0 for n in shape)
        ):
            raise ValueError("Invalid contract array descriptor")
        size = math.prod(shape)
        if offset + size * np.dtype(dtype).itemsize > len(blob):
            raise ValueError("Contract array exceeds binary bounds")
        arrays.append(
            np.frombuffer(blob, dtype=dtype, count=size, offset=offset).reshape(shape)
        )

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"$array"}:
                index = value["$array"]
                if type(index) is not int or not 0 <= index < len(arrays):
                    raise ValueError("Invalid contract array reference")
                return arrays[index]
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    model = decode(header["model"])
    if model.get("family") not in FAMILIES:
        raise ValueError(f"Unsupported model family {model.get('family')!r}")
    return model, hashlib.sha256(encoded).hexdigest()
