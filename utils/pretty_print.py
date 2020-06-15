"""prints a nicely formatted list of arguments.
"""

import time


def pretty_print_arguments(args: dict):
    """prints a nicely formatted list of arguments.

    Arguments:
        args {dict} -- dict of arguments
    """

    longest_key = max([len(key) for key in vars(args)])

    print('Program was launched with the following arguments:')

    for key, item in vars(args).items():
        print("~ {0:{s}} \t {1}".format(key, item, s=longest_key))

    print('')
    # Wait a bit until program execution continues
    time.sleep(0.1)
