from transformers import AdamW, get_constant_schedule_with_warmup
from datasets import load_metric
import torch
from tqdm import tqdm

from DataloaderHelper import get_train_eval_test_dataloaders

cer_metric = load_metric('cer')

def _compute_cer(pred_ids, label_ids, processor, start, print_num_samples):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    if start < print_num_samples:
        print(pred_str)
        print(label_str)
    return cer

def _evaluate_cer(model, dataloader, processor, cer_type, print_num_samples):
    model.eval()
    total_cer = 0.0
    start = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to("cuda:0"))
            # compute metrics
            cer = _compute_cer(outputs, batch["labels"], processor, start, print_num_samples)
            start += 1
            total_cer += cer 

    print(f"{cer_type} CER:", total_cer / len(dataloader))

def enable_gradients(epoch, model):
    i_max = epoch * 4
    i = 0
    for name, param in model.named_parameters():

        if i < i_max or i - 1 < i_max and 'bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

        if 'encoder.embeddings' in name:
            param.requires_grad = True

        if not 'bias' in name:
            i += 1

def run(processor, 
          model, 
          batch_size,
          num_workers,
          image_directory,
          num_epochs, 
          lr,
          save_directory = None,
          save_every = 5,
          print_train_cer_every = 5,
          print_eval_cer_every = 5,
          print_loss_every = 1,
          print_num_samples = 1,
          train = True,
          unfreeze = False,
          use_double = False):
    
    train_dataloader, eval_dataloader, _ = get_train_eval_test_dataloaders(processor, image_directory, batch_size, num_workers, use_double)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, 15000)
    model.cuda()
    losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        if print_eval_cer_every != None and epoch % print_eval_cer_every == 0:
            _evaluate_cer(model, eval_dataloader, processor, "VALIDATION", print_num_samples)

        if print_train_cer_every != None and epoch % print_train_cer_every == 0:
            _evaluate_cer(model, train_dataloader, processor, "TRAIN", print_num_samples)

        if unfreeze:
            enable_gradients(epoch, model)
        
        if train:
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

            if epoch % print_loss_every == 0:
                print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

            losses.append(train_loss / len(train_dataloader))
            
        if save_directory != None and epoch % save_every == 0:
            model.save_pretrained(save_directory + str(epoch))

    if train:
        print(losses)