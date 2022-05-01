from contextlib import contextmanager
from os.path import exists
from os import remove

def create_debugger(log_file):
    def debug(text):
        log_file.write(f"{text}\n\n")

    return debug

@contextmanager
def debug_session(log_filename):
    if exists(log_filename):
        remove(log_filename)

    with open(log_filename, "a") as log_file:
        yield create_debugger(log_file)
