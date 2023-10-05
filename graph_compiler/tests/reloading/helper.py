import torch

pause_logging = True
logged = False


def add_logging():
    global pause_logging

    original = torch.add

    def add_with_logging(*args):
        global logged
        if not pause_logging:
            logged = True

        return original(*args)

    pause_logging = True
    torch.add = add_with_logging


def reset():
    global pause_logging, logged
    pause_logging = True
    logged = False
