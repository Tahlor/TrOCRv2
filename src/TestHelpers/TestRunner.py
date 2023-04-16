from transformers import AdamW, get_constant_schedule_with_warmup
from datasets import load_metric
import torch
from tqdm import tqdm

from transformers import CanineTokenizer, BertTokenizer

from src.TestHelpers.DataloaderHelper import get_train_eval_test_dataloaders
from src.TestHelpers.TestConfiguration import TestConfiguration
from src.PositionEmbeddings.SinusoidalDeitEmbedding import HeightTrOCRProcessor, SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig

cer_metric = load_metric('cer')

def _compute_cer(pred_ids, label_ids, processor, start, print_num_samples):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    if start < print_num_samples:
        try:
            print(pred_str)
            print(label_str)
        except:
            pass
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

def disable_grad(model):
    for name, param in model.named_parameters():
        if name != "encoder.embeddings.position_embeddings" and name != "decoder.model.decoder.embed_positions.weight":
            param.requires_grad = False

def get_model_and_processor(config : TestConfiguration):
    processor = HeightTrOCRProcessor.from_pretrained(config.processor_pretrained_path, height=config.image_height, width=config.image_width)

    if config.tokenizer_type == "CANINE":
        processor.tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
    elif config.tokenizer_type == "BERT":
        processor.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    model_config = SinusoidalVisionEncoderDecoderConfig.from_pretrained(config.model_pretrained_path, 
                                                                        enc_lpe=config.use_learned_position_embeddings, 
                                                                        dec_lpe=config.use_learned_position_embeddings, 
                                                                        image_height=config.image_height, 
                                                                        image_width=config.image_width, 
                                                                        max_length=config.max_tokens)
    
    model = SinusoidalVisionEncoderDecoder.from_pretrained(config.model_pretrained_path, config=model_config, ignore_mismatched_sizes=True)
    
    if config.train_only_embeddings:
        disable_grad(model)
        
    return processor, model

def run(config : TestConfiguration):
    processor, model = get_model_and_processor(config)


    train_dataloader, eval_dataloader, _ = get_train_eval_test_dataloaders(processor, 
                                                                           config.root_directory, 
                                                                           config.batch_size, 
                                                                           config.num_workers, 
                                                                           config.use_double, 
                                                                           config.num_images)
    
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, 500) if config.constant_warmup else None
    model.cuda()
    losses = []
    for epoch in range(config.num_epochs):  # loop over the dataset multiple times

        if config.print_eval_cer_every != None and epoch % config.print_eval_cer_every == 0:
            _evaluate_cer(model, eval_dataloader, processor, "VALIDATION", config.print_num_samples)

        if config.print_train_cer_every != None and epoch % config.print_train_cer_every == 0:
            _evaluate_cer(model, train_dataloader, processor, "TRAIN", config.print_num_samples)

        if config.unfreeze:
            enable_gradients(epoch, model)
        
        if config.train:
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
                if config.constant_warmup:
                    scheduler.step()

            if epoch % config.print_loss_every == 0:
                print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

            losses.append(train_loss / len(train_dataloader))
            
        if config.save_directory != None and epoch % config.save_every == 0:
            model.save_pretrained(config.save_directory + str(epoch))

    if config.train:
        print(losses)