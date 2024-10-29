import os.path

from tqdm import tqdm

bllip_path = '/cw/liir_data/NoCsBack/bllip_87_89_wsj'
# data_path = '/cw/liir_code/NoCsBack/wolf/projects/transformers-struct-guidance/data'
data_path = '/cw/liir_code/NoCsBack/rubenc/transformers-struct-guidance/data'

blip_dataset_type = 'lg'
upper_directories = ['1987', '1988', '1989']
dev_sections = ['1987/w7_001', '1988/w8_001', '1989/w9_010']
test_sections = ['1987/w7_002', '1988/w8_002', '1989/w9_011']

dev_data = []
for section in dev_sections:
    sect_counter = 0
    section_finished = False
    path = os.path.join(bllip_path, section)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        sentences = []
        curr_sent = ''
        for line in lines:
            if line.startswith('(S1'):
                sentences.append(curr_sent)
                curr_sent = line
            else:
                curr_sent = curr_sent + ' ' + line
        sentences = sentences[1:]
        for sent in sentences:
            dev_data.append(sent)
            sect_counter += 1
            if sect_counter >= 500:
                section_finished = True
                break
        if section_finished:
            break
assert len(dev_data) == 1500

test_data = []
for section in test_sections:
    sect_counter = 0
    section_finished = False
    path = os.path.join(bllip_path, section)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        sentences = []
        curr_sent = ''
        for line in lines:
            if line.startswith('(S1'):
                sentences.append(curr_sent)
                curr_sent = line
            else:
                curr_sent = curr_sent + ' ' + line
        sentences = sentences[1:]
        for sent in sentences:
            test_data.append(sent)
            sect_counter += 1
            if sect_counter >= 1000:
                section_finished = True
                break
        if section_finished:
            break
assert len(test_data) == 3000

train_data = []
if blip_dataset_type == 'lg':
    excluded_sections = dev_sections + test_sections
    train_sections = []
    for ud in upper_directories:
        train_sections.extend([os.path.join(ud, name) for name in os.listdir(os.path.join(bllip_path, ud))])
    train_sections = [sect for sect in train_sections if sect not in excluded_sections]

    for section in tqdm(train_sections, desc='preprocessing training data'):
        path = os.path.join(bllip_path, section)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r') as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            sentences = []
            curr_sent = ''
            for line in lines:
                if line.startswith('(S1'):
                    sentences.append(curr_sent)
                    curr_sent = line
                else:
                    curr_sent = curr_sent + ' ' + line
            sentences = sentences[1:]
            for sent in sentences:
                train_data.append(sent)

print('data split: \t{} - {} - {}'.format(len(train_data), len(dev_data), len(test_data)))
with open(os.path.join(data_path, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_data))
with open(os.path.join(data_path, 'dev.txt'), 'w') as f:
    f.write('\n'.join(dev_data))
with open(os.path.join(data_path, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_data))
