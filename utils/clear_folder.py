"""creates temporary folder and ensures that it is empty.
"""

import os


def clear_folder(folder: str):
    """create temporary empty folder.
    If it already exists, all containing files will be removed.

    Arguments:
        folder {[str]} -- Path to the empty folder
    """
    if not os.path.exists(os.path.dirname(folder)):
        os.makedirs(os.path.dirname(folder))

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path) and ('.csv' in file_path or '.ply' in file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
