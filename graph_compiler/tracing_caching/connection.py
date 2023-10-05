import json
import sys
import time

import requests

from tracing_caching.telemetry import obtain_telemetry
from tracing_caching.tracer import trace_obj


headers = {}


class Connector:
    def __init__(self):
        self._user_id = None
        self._api_key = None
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

    def verify_api_key(self, api_key=None):
        if api_key:
            self._api_key = api_key
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

        if not self._token_is_valid():
            self._refresh_token()
        url = f'{self._host_url}/log_telemetry'
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


if __name__ == "__main__":
    key_data = "<your_token>"  # it is read from the file in the main file
    connector = Connector()
    verification_result = connector.verify_api_key(key_data)
    if (verification_result is None):
        sys.exit("Please validate your API TOKEN!")
    else:
        print("API TOKEN is valid!")
        print("User ID: ", verification_result)

    res = connector.log_telemetry()
    print(res)
