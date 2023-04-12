from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from PIL import Image
import torchvision.transforms as transforms

import torch
import pandas as pd
from transformers import get_constant_schedule_with_warmup
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import evaluate
import numpy as np

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df) // 4

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        # print(pixel_values.shape)
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label !=
                  self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(),
                    "labels": torch.tensor(labels)}
        return encoding

class TrainingLoop():

    def __init__(self, model, processor, optimizer, device, lr, scheduler, num_epochs=20):

        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        self.losses_arr = []
        self.cer_validation_arr = []
        self.validation_arr = []

        self.cer_metric = evaluate.load('cer')

    def compute_cer(self, pred_ids, label_ids):
        pred_str = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(
            label_ids, skip_special_tokens=True)
        cer = self.cer_metric.compute(
            predictions=pred_str, references=label_str)
        return cer

    def print_cer(self, dataloader, eval_or_train):
        self.model.eval()
        cers = []
        with torch.no_grad():
            for batch in dataloader:
                # run batch generation
                outputs = self.model.generate(
                    batch["pixel_values"].to(self.device))
                # compute metrics
                cer = self.compute_cer(pred_ids=outputs,
                                        label_ids=batch["labels"])
                cers.append(cer)

        valid_norm = np.sum(cers) / len(cers)
        print(f"{eval_or_train}: ", valid_norm)
        self.model.train()

    def print_eval_cer(self, eval_dataloader):
        self.print_cer(eval_dataloader, "VALIDATION CER")

    def print_train_cer(self, train_dataloader):
        self.print_cer(train_dataloader, "TRAIN CER")

    def print_preds_and_labels(self, batch):
        self.model.eval()
        pred_ids = self.model.generate(batch["pixel_values"].to(self.device))
        pred_str_batch = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = batch["labels"]
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        print("Prediction - Label")
        for i in range(len(pred_str_batch)):
            print(pred_str_batch[i])
            print(label_str[i])
        self.model.train()

    def print_initial_loss(self, train_dataloader):
        train_loss = 0
        for batch in train_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            outputs = self.model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
        print(train_loss/len(train_dataloader))

    def train(self, train_dataloader, eval_dataloader):
        self.print_eval_cer(eval_dataloader)
        self.print_train_cer(train_dataloader)
        # self.print_initial_loss(train_dataloader)

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times
            # train
            train_loss = 0.0
            self.model.train()
            # print("LR: ", self.optimizer)
            tempCounter = 0
        # with train_dataloader as tepoch:
            for batch in train_dataloader: #tepoch:
    
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # if tempCounter < 1:
                    # self.print_preds_and_labels(batch)
                    # tempCounter += 1
                outputs = self.model(**batch)
                # logits = outputs.logits

                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                train_loss += loss.item()
                print(loss.item())

                # loss = outputs.loss
                # self.losses_arr.append(loss.item())

                # tepoch.set_postfix(
                    # loss=loss.item(), child_loss=loss)
                self.scheduler.step()

            print(f"Loss after epoch {epoch}:",
                  train_loss/len(train_dataloader))

            # evaluate

            if epoch % 5 == 0:
                self.model.eval()

                valid_cer = []
                with torch.no_grad():
                    for batch in eval_dataloader:
                        # run batch generation
                        outputs = self.model.generate(
                            batch["pixel_values"].to(self.device))
                        # compute metrics
                        cer = self.compute_cer(pred_ids=outputs,
                                               label_ids=batch["labels"])
                        valid_cer.append(cer)

                valid_norm = np.sum(valid_cer) / len(valid_cer)
                print("Validation CER:", valid_norm)
                self.cer_validation_arr.append(valid_norm)

                self.print_train_cer(train_dataloader)
        self.model.save_pretrained(".")


def double_image_height(im1, im2, color=(256, 256, 256)):
    dst = Image.new('RGB', (max(im1.width, im2.width),
                    im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def double_image_width(im1, im2, color=(256, 256, 256)):
    dst = Image.new('RGB', (im1.width + im2.width,
                    max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def save_image(absolute_path_and_name, processor_image_tensor):
    transform = transforms.ToPILImage()
    changed_image = transform(processor_image_tensor.squeeze(0))
    changed_image.save(absolute_path_and_name)

def get_model_and_proc(height=384, width=384, use_learned_embeddings=True):
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    processor.image_processor.size['height'] = height
    processor.image_processor.size['width'] = width

    config = VisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten')
    
    config.use_learned_position_embeddings = use_learned_embeddings
    config.decoder.use_learned_position_embeddings = use_learned_embeddings
    config.encoder.image_size = (height, width)
    config.max_length = 50

    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=True)
    return processor, model

def trocr_image(model, processor, image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

def inference():
    image1 = Image.open("/home/jesse/trocr/IAM/AllImages/a01-000u-00.jpg").convert("RGB")
    image2 = Image.open("/home/jesse/trocr/IAM/AllImages/a01-000u-01.jpg").convert("RGB")
    doubled_height_image = double_image_height(image1, image2)
    doubled_width_image = double_image_width(image1, image2)

    # processor, model = get_model_and_proc(height=384 * 2, use_learned_embeddings=False)
    # trocr_image(model, processor, doubled_height_image)

    processor, model = get_model_and_proc(use_learned_embeddings=False)
    trocr_image(model, processor, image1)
    # trocr_image(model, processor, doubled_width_image)

    # processor, model = get_model_and_proc(width=384*2, use_learned_embeddings=False)
    # trocr_image(model, processor, doubled_width_image)

def train():

    def set_model_tokens(model, processor):
        model.config.decoder_start_token_id = 2 #processor.tokentizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        model.config.eos_token_id = processor.tokenizer.sep_token_id
        # model.config.max_length = 128
        # model.generation_config.max_length = 128

    def iam_df():
        df = pd.read_fwf('/home/jesse/trocr/IAM/gt_test.txt', header=None)
        df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
        del df[2]
        # some file names end with jp instead of jpg, let's fix this
        df['file_name'] = df['file_name'].apply(
            lambda x: x + 'g' if x.endswith('jp') else x)
        return df

    def new_iam_df(label_directory='/home/jesse/trocr/IAM/train_labels.txt'):
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

    def main():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        # df = iam_df()
        df = new_iam_df()
        print(df.head())
        processor, model = get_model_and_proc(use_learned_embeddings=False)
        model.to(device)

        set_model_tokens(model, processor)

        lr = 5e-5 / (2048 / 12) / 2 #5e-10
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # scheduler = StepLR(optimizer, step_size=1, gamma=1.03)
        scheduler = get_constant_schedule_with_warmup(optimizer, 15000)

        root_dir = '/home/jesse/trocr/IAM/AllImages/'

        # train_df, test_df = train_test_split(df, test_size=0.2)
        # train_df = df
        # test_df = df
        # train_df.reset_index(drop=True, inplace=True)
        # test_df.reset_index(drop=True, inplace=True)

        train_df = new_iam_df()
        test_df = new_iam_df('/home/jesse/trocr/IAM/test_labels.txt')

        train_dataset = IAMDataset(root_dir=(root_dir),
                                df=train_df,
                                processor=processor)
        eval_dataset = IAMDataset(root_dir=(root_dir),
                                df=test_df,
                                processor=processor)

        train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=12)
        eval_dataloader = DataLoader(eval_dataset, batch_size=12, num_workers=12)

        trainingloop = TrainingLoop(model=model, processor=processor, optimizer=optimizer,
                                    device=device, lr=lr, scheduler=scheduler, num_epochs=5000)

        trainingloop.train(train_dataloader, eval_dataloader)

    main()

train()