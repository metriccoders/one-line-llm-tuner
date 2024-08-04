import re
def build_text_files(data_text, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for texts in data_text:
        summary = str(texts).strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)

