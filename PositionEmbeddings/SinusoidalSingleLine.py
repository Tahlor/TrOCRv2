from SinusoidalDeitEmbedding import HeightTrOCRProcessor, SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig
from transformers import AdamW, get_constant_schedule_with_warmup
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
    img_height = 384
    processor = HeightTrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', height=img_height, width=384)
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=False, dec_lpe=False, image_height=img_height, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=False)
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    cer_metric = load_metric("cer")

    train_dataloader, eval_dataloader = get_train_test_dataloaders(processor)
    
    optimizer = AdamW(model.parameters(), lr=5e-5 / (2048 / batch_size) / 2)
    scheduler = get_constant_schedule_with_warmup(optimizer, 15000)
    model.cuda()
    losses = []
    for epoch in range(50):  # loop over the dataset multiple times

        # evaluate
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

        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs
            for k,v in batch.items():
                batch[k] = v.to("cuda:0")

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            scheduler.step()

        print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
        losses.append(train_loss)
            


        model.save_pretrained(root_directory + "SaveDirectory")
    print(losses)

train()