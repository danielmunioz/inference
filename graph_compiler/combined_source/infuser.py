import os
import fileinput
import re

# Specify the directory to traverse
directory_path = '.'

file_names = ["globals","tracked_var_proxy","numpy_proxy","helpers","source_gen","param","conversion","graph","reloader","wrapping","compiler","transpiler","module","visualisation","cacher", "tracer", "telemetry","cache_db","vmap_helpers"]

replacement_names = ["VII",
                     "VVI",
                     "III",
                     "IIX",
                     "VIV",
                     "IVI",
                     "XIX",
                     "IIV",
                     "XII",
                     "VVV",
                     "XVX",
                     "XXI",
                     "VXI",
                     "IXX",
                     "VIX",
                     "VVX",
                     "IXV",
                     "XVV",
                     "IVV"]

line_num = 22

# Specify the code string to be searched and its replacement
for i in range (len(file_names)):
    code_string = "PyInit_" + file_names[i]

    replacement_string = "PyInit_" + replacement_names[i] # For every file there is corresponding encrypted name

    import_string = "from " + f"graph_compiler.{file_names[i]}" + " import "
    replacement_import_string = "from ." + replacement_names[i] + " import "

    import_string2 = "from " + f"graph_compiler.{file_names[i]}" + " import Graph, LazyGraph"
    replacement_import_string2 = "from " + replacement_names[i] + " import Graph , LazyGraph "

    import_string3 = "from " + "graph_compiler " + f"import {file_names[i]}"
    replacement_import_string3 =  f"import {replacement_names[i]}"

    import_string4 = f"import graph_compiler.{file_names[i]}"
    replacement_import_string4 =  f"import {replacement_names[i]}"

    import_string5 = "from " + f"tracing_caching.{file_names[i]} " + "import "
    replacement_import_string5 =  "import " + f"{replacement_names[i]}."

    import_string10 = f"from graph_compiler import"
    replacement_import_string10 = f"from . {replacement_names[i]} " + "import "

    # ToDo : Fix these hardcoded strings to be more generalized. Currently lazy logic applied. 
    import_string6 = "from tracing_caching.cache_db"
    replacement_import_string6 = "from .XVV "

    import_string7 = "from tracing_caching.tracer"
    replacement_import_string7 = "from .VVX"

    import_string8 = "import transpiler.transpiler as transpiler"
    replacement_import_string8 = "from . import XXI as transpiler"

    import_string9 = "from module.module import _transpile_trainable_module"
    replacement_import_string9  = "from .VXI import _transpile_trainable_module"

    import_string11 = "from tracing_caching.cacher import Cacher"
    replacement_import_string11 = "from .VIX import Cacher"

    import_string12 = "from graph_compiler import globals as glob"
    replacement_import_string12 = "import VII as glob"

    import_string13 = "from graph_compiler import source_gen as sg"
    replacement_import_string13 = "from . import VIV as sg"

    import_string14 = "from graph_compiler import helpers"
    replacement_import_string14 = "import IIX"

    import_string15 = "from graph_compiler.special_ops.vmap_helpers import "
    replacement_import_string15 = "from . IVV import "

    import_string16 = "helpers._get_unique_id" # Add beter name convention
    replacement_import_string16 = "IIX.__get_unique_id"

    import_string17 = "helpers.flatten"
    replacement_import_string17 = "IIX.flatten"

    import_string18 = "from graph_compiler.wrapping import FUNC_TO_PATH"
    replacement_import_string18 = "from .VVV import FUNC_TO_PATH"

    import_string19 = "from graph_compiler import tracked_var_proxy as tvp"
    replacement_import_string19 = "import VVI as tvp"

    # For the vmap helpers
    import_string20 = "from graph_compiler.helpers import _generate_id"
    replacement_import_string20 = "from IIX import *"

    import_string21 = "from graph_compiler import source_gen as sg # infuse"
    replacement_import_string21 = "import VIV as sg"

    import_string22 =  "from graph_compiler.helpers import *"
    replacement_import_string22 = "from .IIX import "

    import_string23 = "module._transpile_trainable_module"
    replacement_import_string23 = "VXI._transpile_trainable_module"

    import_string24 = "from module import module"
    replacement_import_string24 = "import VXI"


    os.system(f"mv {file_names[i]}.py {replacement_names[i]}.pyx")
     
    # Traverse through the directory and its subdirectories
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file has .c extension
            # ToDo bring this logic back for static linking.
            #if file.endswith('.c'):
                # Create the file path
                #file_path = os.path.join(subdir, file)
                
                # Use fileinput module to read the file line by line and replace the code string
                #with fileinput.input(files=(file_path), inplace=True) as f:
                    #for line in f:
                        # Replace the code string with the replacement string using regular expression
                        #line = re.sub(code_string, replacement_string, line)
                        #print(line, end="")

            if file.endswith(".pyx"):
                # Create the file path
                file_path = os.path.join(subdir, file)
                
                # Use fileinput module to read the file line by line and replace the code string
                with fileinput.input(files=(file_path), inplace=True) as f:
                    for line in f:
                        # Replace the code string with the replacement string using regular expression
                        line = re.sub(import_string24,replacement_import_string24,line)
                        line = re.sub(import_string23,replacement_import_string23,line)
                        line = re.sub(import_string22,replacement_import_string22,line)
                        line = re.sub(import_string21,replacement_import_string21,line)
                        line = re.sub(import_string20,replacement_import_string20,line)
                        line = re.sub(import_string19,replacement_import_string19,line)
                        line = re.sub(import_string18,replacement_import_string18,line)
                        line = re.sub(import_string17,replacement_import_string17,line)
                        line = re.sub(import_string16,replacement_import_string16,line)
                        line = re.sub(import_string15,replacement_import_string15,line)
                        line = re.sub(import_string14,replacement_import_string14,line)
                        line = re.sub(import_string13,replacement_import_string13,line)
                        line = re.sub(import_string12,replacement_import_string12,line)

                        line = re.sub(import_string, replacement_import_string, line)
                        line = re.sub(import_string2, replacement_import_string2, line)
                        line = re.sub(import_string3,replacement_import_string3,line)
                        line = re.sub(import_string4 , replacement_import_string4,line)
                        line = re.sub(import_string5,replacement_import_string5,line)
                        line = re.sub(import_string6,replacement_import_string6,line)
                        line = re.sub(import_string7,replacement_import_string7,line)
                        line = re.sub(import_string8,replacement_import_string8,line)
                        line = re.sub(import_string9,replacement_import_string9,line)
                        line = re.sub(import_string10,replacement_import_string10,line)
                        line = re.sub(import_string11,replacement_import_string11,line)
                        print(line, end="")



# Open the file for reading
if not os.path.isfile("flag.txt"):

    with open('XVX.pyx', 'r') as f:
        # Read the contents of the file into memory
        contents = f.readlines()

    code_string = r'''
from .IXV import obtain_telemetry
from .VVX import trace_obj
import sys
import logging
import os
import time

import requests
import json

from google.auth.transport.requests import AuthorizedSession as AuthorizedSession_py
from google.auth.transport.requests import Request as Request_py
from google.oauth2 import service_account as service_account_py

cdef AuthorizedSession = AuthorizedSession_py
del AuthorizedSession_py
cdef Request = Request_py
del Request_py
cdef service_account = service_account_py
del service_account_py

# Check if the IVY_ROOT environment variable is set
if "IVY_ROOT" not in os.environ:

    if 'IVY_ROOT' not in os.environ:
        # traverse backwards through the directory tree, searching for .ivy
        current_dir = os.getcwd()
        ivy_folder = None

        while current_dir != '/':  # Stop at the root directory
            if '.ivy' in os.listdir(current_dir):
                ivy_folder = os.path.join(current_dir, '.ivy')
                break
            current_dir = os.path.dirname(current_dir)

        # Set IVY_ROOT to the full path of the .ivy folder if it was found
        if ivy_folder:
            os.environ['IVY_ROOT'] = ivy_folder
        else:
            # If no .ivy folder was found, create one in the cwd
            ivy_folder = os.path.join(os.getcwd(), '.ivy')
            os.mkdir(ivy_folder)
            os.environ['IVY_ROOT'] = ivy_folder

# Allows to set custom location for .ivy folder
if "IVY_ROOT" in os.environ:
    ivy_folder = os.environ["IVY_ROOT"]

# If the IVY_ROOT environment variable is set, check if it points to a valid .ivy folder
if not os.path.isdir(ivy_folder):
    # If not, raise an exception explaining that the user needs to set it to a valid .ivy folder
    raise Exception("IVY_ROOT environment variable is not set to a valid directory. Please create a hidden folder '.ivy' and set IVY_ROOT to this location to set up your local Ivy environment correctly.")

# If the IVY_ROOT environment variable is set and points to a valid .ivy folder, inform the user about preserving the compiler and transpiler caches across multiple machines
logging.warning("To preserve the compiler and transpiler caches across multiple machines, ensure that the relative path of your projects from the .ivy folder is consistent across all machines. You can do this by adding .ivy to your home folder and placing all projects in the same place relative to the home folder on all machines.")

if os.path.isdir(ivy_folder):
    if os.path.isfile(f"{ivy_folder}/key.pem"):
        pass
    else:
        with open(f'{ivy_folder}/key.pem', 'w') as key_pem:
            pass


with open(f'{ivy_folder}/key.pem', 'r') as key_file_py:
    key_data = key_file_py.readline()


cdef key_file = key_file_py
del key_file_py

headers = {}

class Connector:
    def __init__(self, user_api_key):
        self.user_id = None
        self._api_key = user_api_key
        self._token = None
        self._token_exp = None
        self._host_url = 'https://cloud-db-gateway-94jg94af.ew.gateway.dev'
    
    def _token_is_valid(self):
        return time.time() < self._token_exp
    
    def _refresh_token(self):
        result = self.verify_api_key()
        if result is None:
            # backup: shouldn't reach here
            raise Exception("Please validate your API TOKEN!")

    def verify_api_key(self):
        url = f'{self._host_url}/apikey/{self._api_key}'
        response = requests.request('GET', url, headers=headers)

        if response.status_code == 200:
            verification_result = response.json()
            if (verification_result is not None) and (verification_result['user_id'] is not None):
                self._user_id = verification_result['user_id']
                self._token = verification_result['token']
                self._token_exp = verification_result['exp']
                return self._user_id
        return None

    def log_telemetry(self):
        hostname, os_hardware, time_zone, private_ip, public_ip = obtain_telemetry()
        telemetry = json.dumps({
            "user_id": self._user_id,
            "hostname": hostname,
            "os_hardware": os_hardware,
            "time_zone_date": time_zone,
            "private_ip": private_ip,
            "public_ip": public_ip
        })
        print(telemetry)

        if not self._token_is_valid():
            self._refresh_token()
        url = f'{self._host_url}/telemetry'
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", url, headers=headers, data=telemetry)
        return response.text

    def log_compilation(self, obj, args, kwargs, compile_kwargs):
        _, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str = trace_obj(
            obj, args=args, kwargs=kwargs, compile_kwargs=compile_kwargs)
        compile_telemetry = json.dumps({
            'user_id': self._user_id,
            'code_loc': code_loc,
            'code_line': code_line,
            'func_def': func_def,
            'args_str': args_str,
            'kwargs_str': kwargs_str,
            'compile_kwargs_str': compile_kwargs_str
        })

        if not self._token_is_valid():
            self._refresh_token()
        url = f'{self._host_url}/log_compilation'
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", url, headers=headers, data=compile_telemetry)
        return response.text


def is_colab():
    return 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ

connector = Connector(key_data)
verification_result = connector.verify_api_key()

if is_colab():
    glob.user_authorized = True

if (verification_result is None) and (is_colab()==False) :
    sys.exit("Please sign up for free pilot access here [https://console.unify.ai/], in order to make use of ivy.compile, ivy.transpile and ivy.unify.")

else:
    glob.user_authorized = True

#if key_data not in valid_keys:
    #_PTRACE_TRACEME = 0

    #libc_debugger = ctypes.util.find_library("c")
    #dll_debugger = ctypes.CDLL(libc_debugger)
    #result_dll = dll_debugger.ptrace(_PTRACE_TRACEME, 0, ctypes.c_void_p(1), ctypes.c_void_p(0))
    #if result_dll == -1:
        #print("Debugger decteced! Existing...")
        #sys.exit(1)

    #def trace_func(frame, event, arg):
        # Check if the event is "call" and the frame is for the "ptrace" function
        #if event == "call" and frame.f_code.co_name == "ptrace":
            # Raise an exception to prevent the "ptrace" function from being called
            #print("Use your Debugger with respect to our policy Please")
            #sys.exit(1)

    # Set the trace function for the process
    #sys.settrace(trace_func)


    #del _PTRACE_TRACEME
    #del libc_debugger
    #del dll_debugger
    #del result_dll
    #del trace_func

del key_data # leave no key data behind


Module_key = "106301343e283bd0bbe27081aa23d91e0d3549f773311be6f9a89c6d6be43be5"'''

    with open('III.pyx', 'r') as f:
        # Read the contents of the file into memory
        contents_proxy = f.readlines()

    code_string_proxy = '''
import os

Module_key = "106301343e283bd0bbe27081aa23d91e0d3549f773311be6f9a89c6d6be43be5"

current_key = os.environ.get('IVY_SO_KEY')

if current_key == Module_key:
    pass

else:
    import sys
    sys.exit("!!!!!!")'''

    contents.insert(line_num-1, code_string)

    # Open the file for writing
    with open('XVX.pyx', 'w') as f:
        # Write the modified contents back to the file
        contents = "".join(contents)
        f.write(contents)

    contents_proxy.insert(2-1, code_string_proxy)

    with open('III.pyx', 'w') as f:
        # Write the modified contents back to the file
        contents_proxy = "".join(contents_proxy)
        f.write(contents_proxy)

    with open('flag.txt', 'w') as fp:
        pass

with open('IIV.pyx', 'r') as file:
    file_contents = file.read()

multi_line_import = r"from .IIX import \([\w\s,_]+\)"

multi_line_replacement_import = "from .IIX import *"

modified_contents = re.sub(multi_line_import, multi_line_replacement_import , file_contents)

# Write the modified contents back to the file
with open('IIV.pyx', 'w') as file:
    file.write(modified_contents)
