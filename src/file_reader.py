def read_input_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    return data