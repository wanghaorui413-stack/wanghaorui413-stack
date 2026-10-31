"""仅列出完整本地ZIP成员，用于诊断中断的无压缩官方样例。"""

import struct
from pathlib import Path

path = Path("data/satellite/downloads/ESA_L1C_PB0513.partial")
with path.open("rb") as stream:
    while True:
        header = stream.read(30)
        if len(header) < 30 or header[:4] != b"PK\x03\x04":
            break
        values = struct.unpack("<4s5H3I2H", header)
        name = stream.read(values[-2]).decode()
        stream.read(values[-1])
        size = values[7]
        print(name, size, flush=True)
        if stream.tell() + size > path.stat().st_size:
            break
        if name.endswith("MTD_MSIL1C.xml"):
            payload = stream.read(size)
            if values[3] != 0:
                raise ValueError("成员不是原样存储，不输出元数据")
            (path.parent / "scene_metadata.xml").write_bytes(payload)
        else:
            stream.seek(size, 1)
