from pathlib import Path

source_dir = Path(r'D:\Programming.Data\Braille\Книги Анжелы\Лит чтение_ч2')
out_file = 'all.txt'

files = sorted(source_dir.glob('*.marked.txt'))

with (source_dir/out_file).open('w') as f:
    for fi in files:
        f.write(fi.read_text())
        f.write('\n\n')

print((source_dir/out_file).read_text())
