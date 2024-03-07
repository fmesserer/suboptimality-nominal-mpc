import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stderr(to=os.devnull):
    '''
    usage:
    with suppress_stderr():
        do_something()
    '''
    fd = sys.stderr.fileno()

    def _redirect_stdout(to):
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
            

if __name__ == '__main__':
    print('This should be printed')
    with suppress_stderr():
        print('This should also be printed')
        print('This should not be printed', file=sys.stderr)
