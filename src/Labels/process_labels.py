infile_path = '/home/jesse/Laia/egs/iam/data/original/lines.txt'
outfile_path = '/home/jesse/scratch/labels.txt'
with open(outfile_path, 'w') as outfile:
    with open(infile_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                columns = line.strip().split(' ')
                last_col = ' '.join(columns[8:])
                outfile.write(f"{columns[0]}|{last_col}\n")