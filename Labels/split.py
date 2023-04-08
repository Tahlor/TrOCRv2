img_root_path = '/home/jesse/Laia/egs/iam/data/imgs/lines_h128'
file_labels = {}
with open('/home/jesse/scratch/labels.txt', 'r') as label_file:
    for line in label_file:
        data = line.split('|')
        filename = data[0]
        label = '|'.join(data[1:])
        file_labels[filename] = label

def split_file(split_path, split_label_path):
    with open(split_path, 'r') as train_file:
        with open(split_label_path, 'w') as train_labels:
            for line in train_file:
                reformatted_line = line.replace('\n', '')
                if reformatted_line in file_labels:
                    train_labels.write(f'{reformatted_line}|{file_labels[reformatted_line]}')

split_file('/home/jesse/Laia/egs/iam/data/part/lines/aachen/tr.lst', '/home/jesse/scratch/train_labels.txt')
split_file('/home/jesse/Laia/egs/iam/data/part/lines/aachen/te.lst', '/home/jesse/scratch/test_labels.txt')
split_file('/home/jesse/Laia/egs/iam/data/part/lines/aachen/va.lst', '/home/jesse/scratch/validation_labels.txt')