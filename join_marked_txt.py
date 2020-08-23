from pathlib import Path

source_dir = Path(r'D:\Programming.Data\Braille\My\labeled\ASI\Student_Book\56-61')
out_file = 'all.txt'
write_filename = True

files = sorted(source_dir.glob('*.marked.txt'))

with (source_dir/out_file).open('w') as f:
    for fi in files:
        if write_filename:
            f.write(fi.name)
            f.write('\n\n')
        text = fi.read_text(encoding="utf-8")
        f.write(text)
        f.write('\n\n')

print((source_dir/out_file).read_text())
