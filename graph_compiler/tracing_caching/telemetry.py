import platform
import socket
import time

import requests

from tracing_caching.tracer import (get_total_called_functions,
                                    retrieve_nested_source_raw, trace_obj)


def get_hostname(verbose: bool = False):
    hostname = socket.gethostname()
    if verbose:
        print(f"Hostname: {hostname}")
    return hostname


def get_os_hardware(verbose: bool = False):
    os_hardware = platform.platform()
    if verbose:
        print(f"OS Hardware: {os_hardware}")
    return os_hardware


def get_timezone(verbose: bool = False):
    time_zone = f'{time.tzname[0]}-{time.tzname[1]}'
    if verbose:
        print(f'Time Zone: {time_zone}')
    return time_zone


def get_private_ip(verbose: bool = False):
    try:
        private_ip = (((
            [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                if not ip.startswith("127.")] or
            [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0],
               s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) +
            ["no IP found"])[0])
    except:
        private_ip = None

    if verbose:
        print(f"Private IP Address: {private_ip}")
    return private_ip


def get_public_ip(verbose: bool = False):
    try:
        public_ip = requests.get(
            "https://www.wikipedia.org").headers["X-Client-IP"]
    except:
        public_ip = None

    if verbose:
        print(f"Public IP Address: {public_ip}")
    return public_ip


def get_function_trace(obj, args, kwargs, compile_kwargs):
    obj_call, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str = trace_obj(
        obj, args, kwargs, compile_kwargs)
    return obj_call, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str


def get_nested_function_source(obj):
    source_code = retrieve_nested_source_raw(obj)
    return source_code


def get_called_functions(obj):
    called_funcs = get_total_called_functions(obj)
    return called_funcs


def obtain_telemetry(verbose: bool = False):
    hostname = get_hostname(verbose)
    os_hardware = get_os_hardware(verbose)
    time_zone = get_timezone(verbose)
    private_ip = get_private_ip(verbose)
    public_ip = get_public_ip(verbose)
    return hostname, os_hardware, time_zone, private_ip, public_ip


if __name__ == '__main__':
    obtain_telemetry(verbose=True)
