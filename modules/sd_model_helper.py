import torch
import mmap
import json
import os


def load_file(filename, device):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
    offset = n + 8
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}


DTYPES = {"F32": torch.float32}
device = "cpu"


def create_tensor(storage, info, offset):
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape)

def get_xla_device():
    device = "cpu"
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        print("Using XLA device: ", device)
    except ImportError:
        print("No XLA device found")
        pass
    return f"{device}"