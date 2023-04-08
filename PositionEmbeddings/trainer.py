from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from SinusoidalDeitEmbedding import SinusoidalDeiTEmbeddings, SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig, HeightTrOCRProcessor
from transformers import AdamW
# from IAMDataset import IAMDataset
from IAMDatasetTwoLines import IAMDatasetTwoLines as IAMDataset
from datasets import load_metric
import torch
from sklearn.model_selection import train_test_split

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def get_iam_df():
    df = pd.read_fwf('/home/jesse/trocr/IAM/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
    del df[2]
    # some file names end with jp instead of jpg, let's fix this
    df['file_name'] = df['file_name'].apply(
        lambda x: x + 'g' if x.endswith('jp') else x)
    return df

def get_train_test_dfs():
    df = get_iam_df()
    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df

def get_train_test_dataloaders(processor : HeightTrOCRProcessor):
    train_df, test_df = get_train_test_dfs()
    train_dataset = IAMDataset(root_dir='/home/jesse/trocr/IAM/image/',
                           df=train_df,
                           processor=processor)
    eval_dataset = IAMDataset(root_dir='/home/jesse/trocr/IAM/image/',
                            df=test_df,
                            processor=processor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=7, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=7)

    return train_dataloader, eval_dataloader

def set_tokens(model : SinusoidalVisionEncoderDecoder, processor : HeightTrOCRProcessor):
    # model.config.eos_token_id = processor.tokenizer.sep_token_id
    # model.config.decoder_start_token_id = 2 # some tutorials use processor.tokenizer.cls_token_id here
    # model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = 2 #processor.tokenizer.cls_token_id
    model.config.sep_token_id = processor.tokenizer.sep_token_id
    # model.config.max_length = 64
    # model.config.early_stopping = True
    # model.config.no_repeat_ngram_size = 3
    # model.config.length_penalty = 2.0
    # model.config.num_beams = 4

def compute_cer(pred_ids, label_ids, cer_metric, processor, start):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    if start < 5:
        print(pred_str)
        print(label_str)
    return cer

def disable_grad(model):
    for name, param in model.named_parameters():
        if name != "encoder.embeddings.position_embeddings" and name != "decoder.model.decoder.embed_positions.weight":
            param.requires_grad = False

def train():
    new_height = 384 * 2
    processor = HeightTrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', height=new_height, width=384)
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=True, dec_lpe=True, image_height=new_height, image_width=384, max_length=128)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=True)
    disable_grad(model)
    set_tokens(model, processor)
    cer_metric = load_metric("cer")

    train_dataloader, eval_dataloader = get_train_test_dataloaders(processor)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.cuda()
    for epoch in range(1000):  # loop over the dataset multiple times

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

        print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
            


        model.save_pretrained("/home/jesse/scratch/PositionEmbeddings")

train()