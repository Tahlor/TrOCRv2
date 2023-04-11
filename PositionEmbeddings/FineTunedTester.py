from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW, get_constant_schedule_with_warmup
from datasets import load_metric
from IAMDataset import IAMDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd

def compute_cer(pred_ids, label_ids, cer_metric, processor, start):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    if start < 5:
        print(pred_str)
        print(label_str)
    return cer

root_directory = "/home/jesse/TrOCR/"
batch_size = 18
num_workers = 12
def new_iam_df(label_directory=root_directory + '/IAM/train_labels.txt'):
    rows = []
    with open(label_directory, 'r') as label_file:
        for line in label_file:
            data = line.split('|')
            file_name = f'{data[0]}.jpg'
            label = ' '.join(data[1:]).replace('\n','')
            new_row = {'file_name': file_name, 'text': label}
            rows.append(new_row)
    df = pd.DataFrame(rows, columns=['file_name','text'])
    return df

def get_new_train_test_dfs():
    train_df = new_iam_df(root_directory + 'IAM/train_labels.txt')
    test_df = new_iam_df(root_directory + 'IAM/validation_labels.txt')
    return train_df, test_df

def get_train_test_dataloaders(processor):
    train_df, test_df = get_new_train_test_dfs()
    train_dataset = IAMDataset(root_dir=root_directory + 'IAM/AllImages/',
                           df=train_df,
                           processor=processor)
    eval_dataset = IAMDataset(root_dir=root_directory + 'IAM/AllImages/',
                            df=test_df,
                            processor=processor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, eval_dataloader

def train():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
    model = VisionEncoderDecoderModel.from_pretrained('../Results/results/FineTuning/SaveDirectory')
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    cer_metric = load_metric("cer")

    train_dataloader, eval_dataloader = get_train_test_dataloaders(processor)
    # evaluate
    model.cuda()
    model.eval()
    valid_cer = 0.0
    start = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to("cuda:0"))
            # compute metrics
            cer = compute_cer(outputs, batch["labels"], cer_metric, processor, start)
            start += 1
            valid_cer += cer 

    print("Validation CER:", valid_cer / len(eval_dataloader))

    train_cer = 0.0
    start = 0
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to("cuda:0"))
            # compute metrics
            cer = compute_cer(outputs, batch["labels"], cer_metric, processor, start)
            start += 1
            train_cer += cer 

    print("Train CER:", train_cer / len(train_dataloader))

train()