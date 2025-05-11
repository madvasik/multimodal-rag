import gzip
import json
import os
import shutil

def decompress_json_gz(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json.gz"):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename[:-3]
            output_path = os.path.join(output_dir, output_filename)

            try:
                with gzip.open(input_path, 'rt', encoding='utf-8') as f_in:
                    with open(output_path, 'wt', encoding='utf-8') as f_out:
                         shutil.copyfileobj(f_in, f_out)

                print(f"Распакован '{input_path}' в '{output_path}'")
            except Exception as e:
                print(f"Ошибка при обработке файла {input_path}: {e}")

input_directory = r'index_colqwen_new'
output_directory = r'index_colqwen_new\unpacked_json'

decompress_json_gz(input_directory, output_directory)