def read_input_file(file_path):
    """
    Read input file
    :param file_path:
    :return: content of input file
    """
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    return data