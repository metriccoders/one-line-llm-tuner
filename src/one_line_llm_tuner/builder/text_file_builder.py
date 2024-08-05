import re


def build_text_files(data_text, dest_path):
    """
    Saves input data to destination files
    :param data_text:
    :param dest_path:
    :return: None
    """
    f = open(dest_path, 'w')
    data = ''
    for texts in data_text:
        summary = str(texts).strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)
