# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import List, Optional
import threading
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector.nixl_connector_v2 import (
    NixlChannel,
    NixlObserverInterface,
)
from lmcache.v1.storage_backend.connector.nixl_utils import NixlConfig, NixlRole

logger = init_logger(__name__)


class RecvObjPool:
    def __init__(self, enable_gc: bool, recent_add_threshold: int = 40, recycle_threshold: int = 80):
        self.lock = threading.Lock()
        self._data: dict[CacheEngineKey, MemoryObj] = {}
        self._cnt: dict[CacheEngineKey, int] = {}

        # TODO: Remove the hard-code
        # HACK: have a recycle threshold to avoid the memory leak
        self._recent_added_keys: list[CacheEngineKey] = []
        self._recent_add_threshold = recent_add_threshold  # Keep recent keys (reduced from 80)
        self._recycle_threshold = recycle_threshold  # Trigger GC earlier (reduced from 160)

        self._enable_gc = enable_gc
        if not self._enable_gc:
            logger.warning(
                "GC for receiver is disabled, may lead to memory "
                "leak in non-testing environment"
            )

        # Debug information
        self._dbg_shallow_add = 0
        self._dbg_deep_add = 0
        self._dbg_shallow_remove = 0
        self._dbg_deep_remove = 0
        self._dbg_num_get = 0
        self._dbg_num_success_get = 0
        self._dbg_num_contains = 0
        self._dbg_num_success_contains = 0
        self._dbg_num_gc = 0
        self._dbg_last_report_time = time.time()

    def dbg_report(self):
        return  # Disable debug report for now

        curr_time = time.time()
        if curr_time - self._dbg_last_report_time < 5:
            return
        self._dbg_last_report_time = curr_time

        logger.warning("RecvObjPool Debug Info:")
        logger.warning("  - New add: %d", self._dbg_deep_add)
        logger.warning("  - Redundant add: %d", self._dbg_shallow_add)
        logger.warning("  - Shallow remove: %d", self._dbg_shallow_remove)
        logger.warning("  - Deep remove: %d", self._dbg_deep_remove)
        logger.warning("  - Num get: %d", self._dbg_num_get)
        logger.warning("  - Num success get: %d", self._dbg_num_success_get)
        logger.warning("  - Num contains: %d", self._dbg_num_contains)
        logger.warning("  - Num success contains: %d", self._dbg_num_success_contains)
        logger.warning("  - Current num_objs: %d", len(self._data))
        tot_size = sum([self._data[key].get_size() for key in self._data])
        logger.warning("  - Total size: %.2f GB", tot_size / 1024 / 1024 / 1024)
        logger.warning("  - Number of GC: %d", self._dbg_num_gc)

    def _gc(self, force_aggressive: bool = False):
        if not self._enable_gc:
            return

        logger.warning("In GC! (aggressive=%s)", force_aggressive)
        self._dbg_num_gc += 1
        st = time.perf_counter()
        freed_size = 0
        current_keys = set(self._data.keys())
        
        # In aggressive mode, keep fewer recent keys
        if force_aggressive:
            keep_recent = max(10, self._recent_add_threshold // 4)
            recent_keys = set(self._recent_added_keys[-keep_recent:])
        else:
            recent_keys = set(self._recent_added_keys)
            
        keys_to_evict = current_keys - recent_keys
        for key in keys_to_evict:
            freed_size += self._data[key].get_physical_size()
            self._data.pop(key)
            self._cnt.pop(key)
        ed = time.perf_counter()
        logger.warning(
            "GC in %.4f msec, released %.2f GB memory",
            (ed - st) * 1000,
            freed_size / 1024 / 1024 / 1024,
        )

    def add(self, key: CacheEngineKey, obj: MemoryObj):
        with self.lock:
            # TODO: Get rid of this
            self._recent_added_keys.append(key)
            self._recent_added_keys = self._recent_added_keys[
                -self._recent_add_threshold :
            ]

            if key in self._data:
                self._cnt[key] += 1

                # DEBUG
                self._dbg_shallow_add += 1
            else:
                self._data[key] = obj
                self._cnt[key] = 1

                # DEBUG
                self._dbg_deep_add += 1

            # DEBUG
            self.dbg_report()

    def remove(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if key in self._cnt:
                self._cnt[key] -= 1
                if self._cnt[key] == 0:
                    self._data.pop(key)
                    self._cnt.pop(key)

                    # DEBUG
                    self._dbg_deep_remove += 1
                else:
                    # DEBUG
                    self._dbg_shallow_remove += 1

            self.dbg_report()
            return True

    def contains(self, key: CacheEngineKey) -> bool:
        with self.lock:
            # Trigger GC more aggressively based on memory usage
            if len(self._data) >= self._recycle_threshold:
                self._gc()
            elif len(self._data) >= self._recycle_threshold * 2:
                # Extra aggressive GC if we have too many objects
                self._gc(force_aggressive=True)

            # DEBUG
            ret = key in self._data
            self._dbg_num_contains += 1
            if ret:
                self._dbg_num_success_contains += 1
            self.dbg_report()

            return ret

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.lock:
            # DEBUG
            ret = self._data.get(key, None)
            self._dbg_num_get += 1
            if ret is not None:
                self._dbg_num_success_get += 1
            self.dbg_report()

            return ret

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Pin a key to prevent it from being evicted.
        For now, we just return True as pinning is not critical for NixlBackend.
        """
        # TODO: Implement actual pinning logic if needed
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Unpin a key to allow it to be evicted.
        For now, we just return True as unpinning is not critical for NixlBackend.
        """
        # TODO: Implement actual unpinning logic if needed
        return True


class BasicNixlObserver(NixlObserverInterface):
    """
    Basic implementation of the NixlObserverInterface to handle
    events from NixlChannel.
    """

    def __init__(self, obj_pool: RecvObjPool, enable_clone: bool = True):
        """
        Initialize the BasicNixlObserver.
        
        Args:
            obj_pool: The RecvObjPool to add objects to
            enable_clone: Whether to clone tensors (safer but uses more memory)
        """
        self.obj_pool = obj_pool
        self.enable_clone = enable_clone
        if not enable_clone:
            logger.warning("Tensor cloning disabled - may cause data corruption with high concurrency")

    @_lmcache_nvtx_annotate
    def __call__(
        self,
        keys: list[CacheEngineKey],
        objs: list[MemoryObj],
        is_view: bool = True,
    ):
        """Blocking function to process the received objects

        Args:
          keys: the CacheEngineKeys
          objs: the list of MemoryObj
          is_view: whether the memory objects are the view of the underlying
            transfer buffer  (i.e., whether it will be overwrite by next
            transfer)
        """
        clone_time = 0.0
        add_time = 0.0
        for key, value in zip(keys, objs, strict=False):
            assert value.tensor is not None, "The tensor in the MemoryObj is None."
            if is_view and self.enable_clone:
                # Try to clone, but if OOM, trigger aggressive GC and retry
                st = time.perf_counter()
                try:
                    copied_obj = TensorMemoryObj(value.tensor.clone(), value.metadata)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM during clone, triggering aggressive GC")
                    self.obj_pool._gc(force_aggressive=True)
                    torch.cuda.empty_cache()  # Force PyTorch to release cached memory
                    try:
                        copied_obj = TensorMemoryObj(value.tensor.clone(), value.metadata)
                    except torch.cuda.OutOfMemoryError:
                        logger.error("Still OOM after aggressive GC, using view instead of clone")
                        # Fall back to using the view directly (risky but avoids OOM)
                        copied_obj = value
                ed = time.perf_counter()
                self.obj_pool.add(key, copied_obj)
                ed2 = time.perf_counter()
                clone_time += (ed - st) * 1000
                add_time += (ed2 - ed) * 1000
            else:
                # Either not a view or cloning is disabled
                self.obj_pool.add(key, value)
        logger.debug(
            "Nixl Observer: clone time: %.4f msec, Add time: %.4f msec for %d objects",
            clone_time,
            add_time,
            len(keys),
        )


class NixlBackend(StorageBackendInterface):
    """
    Implementation of the StorageBackendInterface for Nixl.

    Currently, the put is synchronized and blocking, to simplify the
    implementation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(self, nixl_config: NixlConfig):
        """
        Initialize the Nixl storage backend.

        :param dst_device: the device where the blocking retrieved KV is stored,
            could be either "cpu", "cuda", or "cuda:0", "cuda:1", etc.
        """
        super().__init__(dst_device=nixl_config.buffer_device)
        # Use more aggressive GC thresholds for high concurrency
        self._obj_pool = RecvObjPool(
            nixl_config.enable_gc,
            recent_add_threshold=40,  # Reduced from 80
            recycle_threshold=80      # Reduced from 160
        )
        # self._data: dict[CacheEngineKey, MemoryObj] = {}
        # self._data_lock = threading.Lock()

        self._nixl_channel = NixlChannel(nixl_config)

        if nixl_config.role == NixlRole.RECEIVER:
            # Enable cloning by default, but can be disabled via environment variable
            import os
            enable_clone = os.getenv("LMCACHE_NIXL_ENABLE_CLONE", "1") != "0"
            self._nixl_observer = BasicNixlObserver(self._obj_pool, enable_clone=enable_clone)
            self._nixl_channel.register_receive_observer(observer=self._nixl_observer)

        self._registered_keys: list[CacheEngineKey] = []
        self._registered_metadatas: list[MemoryObjMetadata] = []
        self._num_payload_added = 0

    # TODO(Jiayi): handle `pin` smantics
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.

        :param key: The key to check
        :param pin: Whether to pin the object in the backend.

        :return: True if the key exists, False otherwise
        """
        return self._obj_pool.contains(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing submit_put_task tasks.

        :param key: The key to check
        :return: True if the key exists in put tasks, False otherwise
        """
        return False

    def register_put_tasks(
        self,
        keys: list[CacheEngineKey],
        metadatas: list[MemoryObjMetadata],
    ) -> None:
        """
        Register the put tasks to the backend.
        """
        if len(self._registered_keys) > 0:
            raise RuntimeError("The backend has already registered put tasks.")

        self._registered_keys = keys
        self._registered_metadatas = metadatas
        self._nixl_channel.prepare_send(keys=keys, metadatas=metadatas)

    def allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = False,  # Added for compatibility with v3 interface
    ) -> MemoryObj:
        """
        Allocate a zero-copy write object for the given shape and dtype.

        This will be seen as "adding a new payload" to the backend.
        """

        self._num_payload_added += 1

        ret = self._nixl_channel.allocate_for_send(shape=shape, dtype=dtype, fmt=fmt)
        assert ret is not None, "Failed to allocate zero-copy buffer from nixl_channel"
        return ret

    def flush_put_tasks(self) -> None:
        """
        Flush the registered tasks
        """
        assert len(self._registered_keys) > 0, (
            "The backend has not registered put tasks."
        )
        assert self._num_payload_added == len(self._registered_keys), (
            "The number of payloads added is not equal to the number ofregistered keys."
        )

        self._nixl_channel.finish_send()
        self._registered_keys = []
        self._registered_metadatas = []
        self._num_payload_added = 0

    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> Optional[List[Future]]:
        memory_objs_metadatas = [memory_obj.meta for memory_obj in memory_objs]
        self.register_put_tasks(keys, memory_objs_metadatas)
        self.flush_put_tasks()
        return None

    def submit_prefetch_task(self, key: CacheEngineKey) -> bool:
        """
        An async function to get the MemoryObj from the storage backend.

        :param key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        raise NotImplementedError

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.

        :param key: The key of the MemoryObj.

        :return: MemoryObj. None if the key does not exist.
        """
        return self._obj_pool.get(key)

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        raise NotImplementedError

    def remove(self, key: CacheEngineKey, force=True) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        return self._obj_pool.remove(key)

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self._nixl_channel.close()

    def get_underlying_allocator(self) -> MemoryAllocatorInterface:
        """
        Get the underlying allocator from Nixl channel.
        """
        return self._nixl_channel.get_allocator()

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Pin the key in the storage backend.
        For NixlBackend, pinning is handled by the obj_pool.
        """
        return self._obj_pool.pin(key)

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Unpin the key in the storage backend.
        For NixlBackend, unpinning is handled by the obj_pool.
        """
        return self._obj_pool.unpin(key)

    @staticmethod
    def CreateNixlBackend(
        config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata
    ) -> "NixlBackend":
        """
        Create a Nixl backend with the given configuration.

        :param nixl_config: The Nixl configuration.
        :param dst_device: The device where the data is stored.

        :return: A NixlBackend instance.
        """
        # Create the Nixl config
        nixl_config = NixlConfig.from_cache_engine_config(config, metadata)
        # Create the Nixl backend
        backend = NixlBackend(nixl_config)
        return backend
