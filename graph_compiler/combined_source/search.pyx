import platform
import sys

def check_python_version():
    version = sys.version_info
    major = version.major
    minor = version.minor
    micro = version.micro

    return f"{major}.{minor}.{micro}"

pythonv = check_python_version()

# Add grid search logic

def load_file_by_os():
    os_name = platform.system()
    if os_name == 'Windows':
        # Load DLL

    if os_name == "Linux":
        import compiler

load_file_by_os()
