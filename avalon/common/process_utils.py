import psutil


def is_pid_running(pid: int):
    for proc in psutil.process_iter():
        try:
            if proc.pid == pid:
                return proc.status() != "zombie"
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def get_process(pid: int):
    for proc in psutil.process_iter():
        try:
            if proc.pid == pid:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None
