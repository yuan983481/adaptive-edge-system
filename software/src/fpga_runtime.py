import ctypes
import fcntl
import mmap
import os
import struct
import time
from array import array
from pathlib import Path
from typing import Any

import numpy as np

from .hardware_capability import (
    CTRL_REGISTERS,
    FPGA_ERROR_CODES,
    FPGA_EXPECTED_CAPABILITY,
    GAMMA_LUT_BASE,
    GAMMA_LUT_WORDS,
)
from .pipeline_compiler import CompiledPipeline, compile_pipeline
from .schemas import PipelinePlan


class FPGAUnavailableError(RuntimeError):
    pass


class FPGAProtocolError(RuntimeError):
    pass


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError(f"{name} must be int or numeric string, got {value!r}")


# Linux ioctl encoding. dma-proxy.h declares int32_t* as the ioctl type.
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_WRITE = 1
_IOC_READ = 2


def _ioc(direction: int, type_char: str, nr_char: str, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (ord(type_char) << _IOC_TYPESHIFT)
        | (ord(nr_char) << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


_IOCTL_ARG_SIZE = ctypes.sizeof(ctypes.c_void_p)
FINISH_XFER = _ioc(_IOC_WRITE, "a", "a", _IOCTL_ARG_SIZE)
START_XFER = _ioc(_IOC_WRITE, "a", "b", _IOCTL_ARG_SIZE)

PROXY_STATUS = {
    0: "PROXY_NO_ERROR",
    1: "PROXY_BUSY",
    2: "PROXY_TIMEOUT",
    3: "PROXY_ERROR",
}


class _UioMap:
    def __init__(self, fd: int, map_index: int, span: int):
        self.span = int(span)
        self.mm = mmap.mmap(
            fd,
            self.span,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=int(map_index) * mmap.PAGESIZE,
        )

    def read32(self, offset: int) -> int:
        offset = int(offset)
        if offset < 0 or offset + 4 > self.span:
            raise FPGAProtocolError(
                f"UIO read outside map: offset=0x{offset:X}, span=0x{self.span:X}"
            )
        return struct.unpack_from("<I", self.mm, offset)[0]

    def write32(self, offset: int, value: int) -> None:
        offset = int(offset)
        if offset < 0 or offset + 4 > self.span:
            raise FPGAProtocolError(
                f"UIO write outside map: offset=0x{offset:X}, span=0x{self.span:X}"
            )
        struct.pack_into("<I", self.mm, offset, int(value) & 0xFFFFFFFF)

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None


class _DmaProxyChannel:
    """One DMA Proxy character device; this runtime uses channel buffer 0 only."""

    def __init__(self, device: str, buffer_size: int):
        self.device = str(device)
        self.buffer_size = int(buffer_size)

        # struct channel_buffer:
        #   unsigned int buffer[BUFFER_SIZE / 4];
        #   enum proxy_status status;  // 4 bytes on GCC/ARM
        #   unsigned int length;
        #   unsigned int dummy0;
        #   unsigned int dummy1;
        self.status_offset = self.buffer_size
        self.length_offset = self.buffer_size + 4
        self.struct_size = self.buffer_size + 16

        if not Path(self.device).exists():
            raise FPGAUnavailableError(f"DMA Proxy device not found: {self.device}")

        self.fd = os.open(self.device, os.O_RDWR | getattr(os, "O_SYNC", 0))
        self.mm = mmap.mmap(
            self.fd,
            self.struct_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=0,
        )
        self._buffer_id = array("i", [0])

    def set_length(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        if nbytes <= 0 or nbytes > self.buffer_size:
            raise FPGAProtocolError(
                f"DMA transfer size {nbytes} exceeds proxy buffer size {self.buffer_size}"
            )
        struct.pack_into("<I", self.mm, self.length_offset, nbytes)

    def write_data(self, data) -> None:
        nbytes = len(data)
        if nbytes > self.buffer_size:
            raise FPGAProtocolError(
                f"DMA input {nbytes} bytes exceeds proxy buffer size {self.buffer_size}"
            )
        self.mm[0:nbytes] = data

    def clear_data(self, nbytes: int) -> None:
        self.mm[0:int(nbytes)] = b"\x00" * int(nbytes)

    def read_data(self, nbytes: int) -> bytes:
        return self.mm[0:int(nbytes)]

    def status(self) -> int:
        return struct.unpack_from("<I", self.mm, self.status_offset)[0]

    def start(self) -> None:
        fcntl.ioctl(self.fd, START_XFER, self._buffer_id, True)

    def finish(self) -> None:
        fcntl.ioctl(self.fd, FINISH_XFER, self._buffer_id, True)
        status = self.status()
        if status != 0:
            raise FPGAProtocolError(
                f"DMA Proxy transfer failed on {self.device}: "
                f"{PROXY_STATUS.get(status, f'UNKNOWN_{status}')}"
            )

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class EdfDmaProxyRuntime:
    """AMD EDF runtime: UIO AXI-Lite + Linux DMA Proxy.

    Data:
        /dev/dma_proxy_tx -> AXI DMA MM2S -> edge_accel
        edge_accel -> AXI DMA S2MM -> /dev/dma_proxy_rx

    Control:
        /dev/uioX map0 -> edge_accel CTRL AXI-Lite
        /dev/uioX map1 -> Gamma LUT AXI-Lite
    """

    def __init__(self, cfg: dict):
        fpga = cfg["fpga"]
        self.timeout_s = float(fpga.get("timeout_s", 5.0))
        self.expected_capability = _parse_int(
            fpga.get("expected_capability", FPGA_EXPECTED_CAPABILITY),
            "fpga.expected_capability",
        )

        self.ctrl_reg = dict(CTRL_REGISTERS)
        for name, value in (fpga.get("registers") or {}).items():
            if name not in self.ctrl_reg:
                raise ValueError(f"unknown FPGA register override: {name}")
            self.ctrl_reg[name] = _parse_int(value, f"fpga.registers.{name}")

        self.uio_device = str(fpga.get("uio_device", "/dev/uio0"))
        self.ctrl_map_index = int(fpga.get("ctrl_uio_map", 0))
        self.gamma_map_index = int(fpga.get("gamma_uio_map", 1))
        self.mmio_span = _parse_int(
            fpga.get("mmio_span", 0x10000), "fpga.mmio_span"
        )
        self.dma_buffer_size = _parse_int(
            fpga.get("dma_buffer_size", 512 * 1024),
            "fpga.dma_buffer_size",
        )
        self.tx_device = str(
            fpga.get("dma_tx_device", "/dev/dma_proxy_tx")
        )
        self.rx_device = str(
            fpga.get("dma_rx_device", "/dev/dma_proxy_rx")
        )

        if not Path(self.uio_device).exists():
            raise FPGAUnavailableError(f"UIO device not found: {self.uio_device}")

        self._uio_fd = os.open(
            self.uio_device, os.O_RDWR | getattr(os, "O_SYNC", 0)
        )
        self.ctrl_mmio = None
        self.gamma_mmio = None
        self.tx = None
        self.rx = None

        try:
            self.ctrl_mmio = _UioMap(
                self._uio_fd, self.ctrl_map_index, self.mmio_span
            )
            self.gamma_mmio = _UioMap(
                self._uio_fd, self.gamma_map_index, self.mmio_span
            )
            self.tx = _DmaProxyChannel(
                self.tx_device, self.dma_buffer_size
            )
            self.rx = _DmaProxyChannel(
                self.rx_device, self.dma_buffer_size
            )
        except Exception:
            self.close()
            raise

        self._capability_verified = False
        self._try_verify_capability()

    def _ctrl_write(self, name: str, value: int) -> None:
        self.ctrl_mmio.write32(self.ctrl_reg[name], value)

    def _ctrl_read(self, name: str) -> int:
        return self.ctrl_mmio.read32(self.ctrl_reg[name])

    def _wait_vld(self, ctrl_name: str, label: str) -> None:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self._ctrl_read(ctrl_name) & 0x1:
                return
            time.sleep(0.0005)
        raise TimeoutError(f"timeout waiting for {label}_ap_vld")

    def _verify_capability_value(self, actual: int) -> None:
        expected = self.expected_capability
        actual_major = (actual >> 24) & 0xFF
        expected_major = (expected >> 24) & 0xFF
        if actual_major != expected_major:
            raise FPGAProtocolError(
                f"FPGA contract major mismatch: expected {expected_major}, "
                f"got {actual_major}; capability=0x{actual:08X}"
            )

        expected_mask = expected & 0xFFFF
        actual_mask = actual & 0xFFFF
        missing = expected_mask & ~actual_mask
        if missing:
            raise FPGAProtocolError(
                f"FPGA capability is missing required operators: "
                f"missing_mask=0x{missing:04X}, actual=0x{actual:08X}, "
                f"expected=0x{expected:08X}"
            )
        self._capability_verified = True

    def _try_verify_capability(self) -> bool:
        if self._ctrl_read("capability_ctrl") & 0x1:
            self._verify_capability_value(
                self._ctrl_read("capability")
            )
            return True
        return False

    def _verify_capability_after_run(self) -> None:
        if self._capability_verified:
            return
        self._wait_vld("capability_ctrl", "capability")
        self._verify_capability_value(
            self._ctrl_read("capability")
        )

    def _program_gamma_lut(self, compiled: CompiledPipeline) -> None:
        words = compiled.gamma_lut_words
        if words is None:
            return
        if len(words) != GAMMA_LUT_WORDS:
            raise FPGAProtocolError(
                f"Gamma LUT has {len(words)} words; "
                f"expected {GAMMA_LUT_WORDS}"
            )
        for index, word in enumerate(words):
            self.gamma_mmio.write32(
                GAMMA_LUT_BASE + index * 4,
                int(word) & 0xFFFFFFFF,
            )

    def _program_registers(self, compiled: CompiledPipeline) -> None:
        print()
        print("=== FPGA COMPILED REGISTERS ===")

        for name, value in compiled.registers.items():
            if name not in self.ctrl_reg:
                raise FPGAProtocolError(
                    f"unknown CTRL register: {name}"
                )

            addr = self.ctrl_reg[name]
            self._ctrl_write(name, value)
            readback = self._ctrl_read(name)

            print(
                f"{name:12s} "
                f"addr=0x{addr:03X} "
                f"write=0x{int(value):08X} "
                f"read=0x{int(readback):08X}"
            )

        self._program_gamma_lut(compiled)

    def _start(self) -> None:
        self._ctrl_write("ctrl", 0x1)

    def _wait_done(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            ctrl = self._ctrl_read("ctrl")
            if ctrl & 0x2:
                return
            time.sleep(0.0005)
        raise TimeoutError(
            f"FPGA AP_DONE timeout after {self.timeout_s:.3f}s"
        )

    def _check_status(self) -> None:
        self._wait_vld("status_ctrl", "status")
        status = self._ctrl_read("status")
        if status & 0x1:
            code = (status >> 4) & 0xF
            label = FPGA_ERROR_CODES.get(
                code, f"UNKNOWN_{code}"
            )
            raise FPGAProtocolError(
                f"FPGA rejected pipeline: {label} "
                f"(status=0x{status:08X})"
            )

    def configure(
        self, plan: PipelinePlan, rows: int, cols: int
    ) -> CompiledPipeline:
        compiled = compile_pipeline(
            plan, rows=rows, cols=cols
        )
        self._program_registers(compiled)
        return compiled

    def _transfer_gray_configured(self, gray_u8: np.ndarray) -> np.ndarray:
        """
        Transfer one already-configured frame through the FPGA.

        Caller must configure rows/cols before calling this function.
        """
        gray = np.ascontiguousarray(gray_u8, dtype=np.uint8)

        if gray.ndim != 2:
            raise ValueError(
                f"FPGA expects a 2-D GRAY8 image, got shape={gray.shape}"
            )

        rows, cols = gray.shape
        nbytes = rows * cols

        if nbytes > self.dma_buffer_size:
            raise FPGAProtocolError(
                f"frame {cols}x{rows} = {nbytes} bytes exceeds "
                f"DMA Proxy buffer {self.dma_buffer_size} bytes"
            )

        self.tx.write_data(memoryview(gray).cast("B"))
        self.rx.clear_data(nbytes)

        self.tx.set_length(nbytes)
        self.rx.set_length(nbytes)

        # Queue receiver first so FPGA output always has somewhere to go.
        self.rx.start()
        self.tx.start()
        self._start()

        self.tx.finish()
        self.rx.finish()

        self._wait_done()
        self._check_status()
        self._verify_capability_after_run()

        result = np.frombuffer(
            self.rx.read_data(nbytes),
            dtype=np.uint8
        ).copy()

        return result.reshape(rows, cols)


    def process_gray(
        self, gray_u8: np.ndarray, plan: PipelinePlan
    ) -> np.ndarray:
        """
        Execute one automatically planned FPGA pipeline.

        Frames larger than the currently proven-safe 4096-pixel FPGA
        transaction are processed as overlapping 64x64 tiles.

        The analyzer/planner still runs once for the whole image. Every tile
        uses the same PipelinePlan, so automatic algorithm selection remains
        global and consistent across the frame.
        """
        if gray_u8.ndim != 2:
            raise ValueError(
                f"FPGA expects a 2-D GRAY8 image, "
                f"got shape={gray_u8.shape}"
            )

        gray = np.ascontiguousarray(gray_u8, dtype=np.uint8)
        rows, cols = gray.shape
        nbytes = rows * cols

        # Current FPGA image path has been verified at 64x64 = 4096 pixels.
        # Keep small images on the original single-transfer path.
        if nbytes <= 4096:
            self.configure(plan, rows=rows, cols=cols)
            return self._transfer_gray_configured(gray)

        TILE = 64
        HALO = 4
        CORE = TILE - 2 * HALO

        if CORE <= 0:
            raise FPGAProtocolError("invalid FPGA tiling geometry")

        print()
        print("=== FPGA TILED EXECUTION ===")
        print(f"frame={cols}x{rows}")
        print(f"tile={TILE}x{TILE}")
        print(f"halo={HALO}")
        print(f"core={CORE}x{CORE}")

        # Configure once. All FPGA transactions below are fixed 64x64.
        # The same automatically generated plan is used for every tile.
        self.configure(plan, rows=TILE, cols=TILE)

        # Pad the whole image once. Using edge replication gives neighborhood
        # operators valid context at the outer image boundary.
        padded = np.pad(
            gray,
            ((HALO, HALO), (HALO, HALO)),
            mode="edge",
        )

        output = np.empty_like(gray)

        tile_count = 0

        for y0 in range(0, rows, CORE):
            core_h = min(CORE, rows - y0)

            for x0 in range(0, cols, CORE):
                core_w = min(CORE, cols - x0)

                # Because the complete image was padded by HALO, starting at
                # (y0,x0) here automatically includes HALO pixels above/left.
                patch = padded[
                    y0:y0 + core_h + 2 * HALO,
                    x0:x0 + core_w + 2 * HALO,
                ]

                ph, pw = patch.shape

                # Bottom/right edge tiles can be smaller than 64x64.
                # Pad them to the exact transaction size already proven safe.
                if ph < TILE or pw < TILE:
                    patch = np.pad(
                        patch,
                        ((0, TILE - ph), (0, TILE - pw)),
                        mode="edge",
                    )

                if patch.shape != (TILE, TILE):
                    raise FPGAProtocolError(
                        f"internal tile shape error: {patch.shape}"
                    )

                tile_result = self._transfer_gray_configured(patch)

                # Throw away the overlap. Only the center belongs to this tile.
                output[
                    y0:y0 + core_h,
                    x0:x0 + core_w,
                ] = tile_result[
                    HALO:HALO + core_h,
                    HALO:HALO + core_w,
                ]

                tile_count += 1

        print(f"tiles_processed={tile_count}")
        print("=== FPGA TILED EXECUTION COMPLETE ===")
        print()

        return output

    def close(self) -> None:
        for name in ("tx", "rx", "ctrl_mmio", "gamma_mmio"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, name, None)

        fd = getattr(self, "_uio_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._uio_fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
