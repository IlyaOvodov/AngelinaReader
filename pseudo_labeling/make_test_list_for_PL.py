"""
create file train.txt from processed list of samples (res.csv produced by data_statistics.py)
"""
from pathlib import Path
import local_config
data_path = Path(local_config.data_path)

folder = 'arch211010'
data_folder = data_path / f'../braille/v1/{folder}'
test_file_list_file = data_path / 'AngelinaDataset/uploaded/all_selected_test_images_list.txt'
out_file = data_path / f'{folder}.train.txt'


test_files = set()
with open(test_file_list_file) as f:
    for line in f:
        test_files.add(line.strip())

total = 0
test_total = 0
found_test_files = set()
with open(out_file, 'w') as outf:
    with open(data_folder / 'res.csv') as f:
        for line in f:
            if line.startswith('name; user; lang; is_public;') or not line.strip():
                continue
            fields = line.split(';')
            fn = fields[0].strip()
            lang = fields[2].strip()
            is_public = fields[3].strip()
            assert is_public in ('True', 'False'), line
            if not '(dup)' in fn and lang == 'RU' and is_public == 'True':
                if fn in test_files:
                    test_total += 1
                    found_test_files.add(fn)
                    print('found int test list: ', fn)
                    continue
                total += 1
                full_fn = 'results/'+fn+'.labeled.jpg'
                assert (data_folder / full_fn).is_file()
                outf.write(full_fn + '\n')
print('total:', total, ' test_total: ', test_total)

for fn in test_files:
    if fn not in found_test_files:
        print('not found: ', fn)
