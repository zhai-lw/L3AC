import contextlib
import inspect
import math
import os
import pathlib
import time
import datetime
from typing import Union, Iterable, Sequence

import logging

import numpy

from . import output
from . import file
from . import module
from .struct.tree import Tree

LOG2Console = True
LOG_FILE = None
LOG_LEVEL = logging.INFO

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

REFRESH_PER_SECOND = 1


def get_logger(logger_name=None, log2console=None, log_file=None, log_level=None):
    if logger_name is None:
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        logger_name = f"{mod.__name__}-log"
    if log2console is None:
        log2console = LOG2Console
    if log_file is None:
        log_file = LOG_FILE
    if log_level is None:
        log_level = LOG_LEVEL
    if logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.warning("logger already exists, please check the logger name")
        return logger
    else:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        if log2console:
            logger.addHandler(get_console_handler())
        if log_file:
            logger.addHandler(get_file_handler(log_file))
        return logger


# log = get_logger()
# log.debug("log debug test")
# log.info("log info test")
# log.warning("log warning test")
# log.error("log error test")
# log.exception("log exception test")
# log.critical("log critical test")


if module.installed('rich'):
    import rich.progress
    import rich.text
    import rich.logging

    from rich.traceback import install

    install(show_locals=True, suppress=['torch'], locals_max_length=4, locals_max_string=50)


    class RichProgressBar:
        class SpeedColumn(rich.progress.ProgressColumn):
            max_refresh = 1

            def render(self, task):
                speed = task.finished_speed or task.speed
                if speed is None:
                    return rich.text.Text("?", style="progress.data.speed")
                return rich.text.Text("{:.3f}/s".format(speed), style="progress.data.speed")

        class TimeColumn(rich.progress.ProgressColumn):
            max_refresh = 1

            def render(self, task):
                result_str = " {}/{} ".format(task.completed, task.total)

                result_str += ' time '
                if task.start_time is None:
                    result_str += '(-:--:--)'
                else:
                    used = task.get_time() - task.start_time
                    result_str += f'({datetime.timedelta(seconds=int(used))})'

                remaining = task.time_remaining
                if remaining is None:
                    result_str += "|(-:--:--)"
                else:
                    result_str += f'|({datetime.timedelta(seconds=int(remaining))})'
                return rich.text.Text(result_str, style="progress.remaining")

        class SuffixColumn(rich.progress.ProgressColumn):
            max_refresh = 1

            def render(self, task):
                if 'suffix' in task.fields:
                    return rich.text.Text(" " + task.fields["suffix"], style="progress.suffix")
                return rich.text.Text("", style="progress.suffix")

        def __init__(self):
            self.progress = rich.progress.Progress(
                rich.progress.TextColumn("[progress.description]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                RichProgressBar.SpeedColumn(),
                RichProgressBar.TimeColumn(),
                RichProgressBar.SuffixColumn(),
                transient=False,
                refresh_per_second=REFRESH_PER_SECOND)
            self.progress.start()

        def progressbar(self, sequence: Union[Iterable, Sequence], length=None,
                        desc: str = 'working'):
            task_id = self.progress.add_task(desc, transient=False)
            progress_sequence = self.progress.track(sequence, task_id=task_id, total=length)
            for item in progress_sequence:
                yield item
            self.progress.remove_task(task_id=task_id)


    default_rich_progress = RichProgressBar()
    progressbar = default_rich_progress.progressbar

    out_fun = default_rich_progress.progress.print


    def get_console_handler():
        return rich.logging.RichHandler(
            rich_tracebacks=True,
            # omit_repeated_times=False,
            # show_level=False,
            # show_path=False,
            # show_time=False,
        )


else:

    from tqdm import tqdm


    def progressbar(*args, length=None, **kwargs):
        if length:
            length = math.ceil(length)
        return tqdm(*args, total=length, disable=not LOG2Console, **kwargs)


    out_fun = tqdm.write


    class ConsoleLogHandler(logging.Handler):

        class LogFormatter(logging.Formatter):
            grey = "\x1b[38;20m"
            yellow = "\x1b[33;20m"
            red = "\x1b[31;20m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"

            FORMATS = {
                logging.DEBUG: grey + LOG_FORMAT + reset,
                logging.INFO: grey + LOG_FORMAT + reset,
                logging.WARNING: yellow + LOG_FORMAT + reset,
                logging.ERROR: red + LOG_FORMAT + reset,
                logging.CRITICAL: bold_red + LOG_FORMAT + reset
            }

            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt)
                return formatter.format(record)

        def __init__(self):
            super().__init__()
            self.setFormatter(ConsoleLogHandler.LogFormatter())

        def emit(self, record):
            try:
                msg = self.format(record)
                out_fun(msg)
                self.flush()
            except RecursionError:
                raise
            except Exception:
                self.handleError(record)
                raise


    def get_console_handler():
        return ConsoleLogHandler()


def get_file_handler(log_file=LOG_FILE):
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return file_handler


# import contextlib
# import inspect
#
#
# @contextlib.contextmanager
# def redirect_to_tqdm():
#     # Store builtin print
#     old_print = print
#
#     def new_print(*args, **kwargs):
#         try:
#             tqdm.write(*args, **kwargs)
#         except TqdmTypeError:
#             old_print(*args, **kwargs)
#
#     try:
#         inspect.builtins.print = new_print
#         yield
#     finally:
#         inspect.builtins.print = old_print


class FuncLogger:
    def __init__(self):
        self.log_his = Tree((0, 0))
        self.current_func = []

    def add2func_his(self, fun, used_time):
        avg_time, times = self.log_his[fun]
        self.log_his[fun] = numpy.average((avg_time, used_time), weights=(times, 1)), times + 1

    def add(self, func):
        def wrapper(*args, **kwargs):
            if len(args) == 0:
                func_name = '{func}()'.format(func=func.__name__)
            else:
                func_name = '{cname}.{func}()'.format(cname=file.fullname(args[0]), func=func.__name__)
            self.current_func.append(func_name)
            func_result, used_time = func_time_cal(func, args, kwargs)
            self.add2func_his(self.current_func, used_time)
            # print('->'.join(self.current_func), 'used {:.5f}s'.format(used_time))
            self.current_func.pop()
            return func_result

        return wrapper

    def get_result(self):
        result_dict = {}
        total_time = 0
        for func_name, (avg_time, call_times) in self.log_his.items(deep=1):
            total_time += avg_time * call_times
        result_dict['total'] = 'total used {:.5f}'.format(total_time)

        for func_name, (avg_time, call_times) in self.log_his.items():
            result_dict['->'.join(func_name)] = 'average time {:.5f}s\t total {:.2f}s\t {} times' \
                .format(avg_time, avg_time * call_times, call_times)

        return result_dict


def func_time_cal(func, args, kwargs):
    start_time = time.time()
    func_result = func(*args, **kwargs)
    used_time = time.time() - start_time
    return func_result, used_time


default_func_logger = FuncLogger()

add_func_logger = default_func_logger.add

get_func_log = default_func_logger.get_result


def print_func_log(func_logger=None, print_func=print):
    if func_logger is None:
        func_logger = default_func_logger
    output.dictionary(func_logger.get_result(), revert=True, out_fun=print_func)


noprint = lambda: contextlib.redirect_stdout(open(os.devnull, 'w'))


class TimeRecorder:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cost_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.cost_time = self.end_time - self.start_time


def release_source(release_dir: pathlib.Path, source_path: pathlib.Path = None, as_zip=True):
    if source_path is None:
        source_path = file.SOURCE_PATH

    import shutil
    if as_zip:
        shutil.make_archive(str(release_dir / "source"), 'zip', source_path)
    else:
        shutil.copytree(source_path, (release_dir / "source"), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '*.pyo', '__pycache__'))
