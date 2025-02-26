def get_my_python_version():
    return "3.10"

def get_my_python_tag():
    return "cp310"

def get_my_python_copts():
    copts = [
        "-DCPP_PYVENV_LAUNCHER=\\\"/usr/bin/python{}\\\"".format(get_my_python_version()),
        "-DCPP_PYTHON_HOME=\\\"/usr\\\"",  # Or wherever your Python is installed
        "-DCPP_PYTHON_PATH=\\\"/usr/lib/python{}/\\\"".format(get_my_python_version()),
    ]
    return copts