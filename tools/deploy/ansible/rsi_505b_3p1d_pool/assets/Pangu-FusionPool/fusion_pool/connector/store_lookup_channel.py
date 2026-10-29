# SPDX-License-Identifier: Apache-2.0
"""PFC lookup channel. Bodies follow native LookupKeyClient / LookupKeyServer."""

import os
import threading
from contextlib import contextmanager

import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker as native
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import BlobBlockHashes

from fusion_pool.connector.capacity_only import is_capacity_only


class HBMLookupKeyClient(native.LookupKeyClient):
    """ZMQ client for the LookupKey admin channel.

    Routes both prefix-cache lookups and admin commands (currently:
    ``reset``) to ``LookupKeyServer`` on worker rank 0. The first frame
    of every request is a named tag from ``protocol.py``.
    """

    def _lookup(
        self,
        token_len,
        block_hashes,
        hbm_hit_tokens=0,
    ):
        hash_len = len(block_hashes[0]) if block_hashes else 0
        all_frames = (
            native.LOOKUP_MSG,
            token_len.to_bytes(4, byteorder="big"),
            hash_len.to_bytes(2, byteorder="big"),
            b"".join(block_hashes),
            hbm_hit_tokens.to_bytes(4, byteorder="big"),
        )
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        return int.from_bytes(resp, "big")

    def lookup(self, req_id, token_len, block_hashes, non_block=False, *, hbm_hit_tokens=0):
        """Return None until a non-blocking lookup is ready, so the caller retries later."""
        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(
                self._lookup,
                token_len,
                list(block_hashes),
                hbm_hit_tokens,
            )
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            native.logger.error("Async Mooncake lookup failed for %s: %s", req_id, e)
            return 0
        finally:
            del self.futures[req_id]


class HBMLookupKeyServer(native.LookupKeyServer):
    """ZMQ server on worker rank 0 for the LookupKey admin channel.

    Handles two request types, tagged at frame 0:
    - ``LOOKUP_MSG``: prefix-cache hit query, returns hit count.
    - ``RESET_MSG``: drains the send thread queue, then runs
      ``store.remove_all(force=True)``. Caller must have paused the
      scheduler first.
    """

    def __init__(self, store_worker, vllm_config):
        self.ctx = zmq.Context()
        socket_path = native.get_zmq_rpc_path_lookup(vllm_config)
        self._ipc_path = socket_path.removeprefix("ipc://")
        if os.path.exists(self._ipc_path):
            os.unlink(self._ipc_path)
        self.socket = native.make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,
            bind=True,
        )

        self.store_worker = store_worker
        self.running = True

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                msg_type = bytes(all_frames[0])

                if msg_type == native.LOOKUP_MSG:
                    token_len = int.from_bytes(all_frames[1], byteorder="big")
                    hash_len = int.from_bytes(all_frames[2], byteorder="big")
                    blob = all_frames[3].buffer
                    hbm_hit_tokens = int.from_bytes(all_frames[4], byteorder="big")
                    block_hashes = BlobBlockHashes(blob, hash_len)
                    result = self.store_worker.lookup(
                        token_len,
                        block_hashes,
                        hbm_hit_tokens=hbm_hit_tokens,
                    )
                    self.socket.send(result.to_bytes(4, byteorder="big"))

                elif msg_type == native.RESET_MSG:
                    try:
                        # Drain in-flight puts before wiping the master;
                        # otherwise stale puts can repopulate it post-reset.
                        # Safe across HMA: store.remove_all wipes the underlying
                        # flat key space, clearing every (group_id, hash) entry.
                        if self.store_worker.kv_send_thread is not None:
                            self.store_worker.kv_send_thread.request_queue.join()
                        self.store_worker.store.remove_all(force=True)
                        native.logger.info(
                            "Mooncake store reset via remove_all succeeded."
                        )
                        self.socket.send(native.RESP_OK)
                    except Exception as e:
                        native.logger.error("Mooncake remove_all failed: %s", e)
                        self.socket.send(native.RESP_ERR)

                else:
                    native.logger.warning(
                        "LookupKeyServer received unknown msg_type: %r",
                        msg_type,
                    )
                    self.socket.send(native.RESP_ERR)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()


@contextmanager
def hbm_lookup_channel(config, side):
    """Install the patched lookup channel for this PFC instance."""
    if is_capacity_only(config):
        yield
        return
    if side == "worker":
        module, name, replacement = native, "LookupKeyServer", HBMLookupKeyServer
    else:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import scheduler
        module, name, replacement = scheduler, "LookupKeyClient", HBMLookupKeyClient
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)
