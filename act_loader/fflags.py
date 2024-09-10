from dataclasses import dataclass
import mmap


@dataclass
class FeatureFlags:
    set_file_handle_limits: bool = True
    log_bin_data: bool = True
    log_final_bin_shapes: bool = False
    log_info: bool = False
    record_timing: bool = False
    use_async_mmap: bool = True
    use_madv_sequential: bool = True
    use_madv_dontneed: bool = True
    use_madv_hugepage: bool = True
    use_map_private: bool = True
    sync_flushes: bool = True
    pages_per_flush: int = 128  # ignored when use_madv_hugepage is used

    @property
    def mmap_flags(self):
        if self.use_map_private:
            return mmap.MAP_PRIVATE
        else:
            return mmap.MAP_SHARED
