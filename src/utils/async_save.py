import torch, multiprocessing, threading, os
    
def _proc_save(state, tmp_path):
    # Do the blocking torch.save in a separate process
    torch.save(state, tmp_path)

def async_torch_save(state, dest_path):
    """
    Save `state` asynchronously to `dest_path`.
    Writes to a .saving temp file then atomically renames.
    """
    tmp_path = dest_path + ".saving"
    p = multiprocessing.Process(target=_proc_save, args=(state, tmp_path))
    p.start()

    # background thread waits for save then renames
    def _finalize():
        p.join()
        if os.path.exists(tmp_path):
            os.replace(tmp_path, dest_path)
    threading.Thread(target=_finalize, daemon=True).start()

    return p.pid  # optional: you can log pid
