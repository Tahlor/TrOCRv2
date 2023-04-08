from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, TrOCRProcessor, PretrainedConfig
from transformers.models.deit.modeling_deit import DeiTEmbeddings, DeiTPatchEmbeddings, DeiTConfig

import os
import math
import torch
from typing import Optional, Union
from torch import nn

from PIL import Image

class SinusoidalVisionEncoderDecoder(VisionEncoderDecoderModel):
    def __init__(self, config):
        super().__init__(config)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        config = kwargs.get('config')
        model.encoder.embeddings = SinusoidalDeiTEmbeddings(model.encoder.embeddings, config)
        return model

class SinusoidalVisionEncoderDecoderConfig(VisionEncoderDecoderConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.encoder.use_learned_position_embeddings = kwargs.get("enc_lpe", True)
        config.decoder.use_learned_position_embeddings = kwargs.get("dec_lpe", True)
        height = kwargs.get("image_height", 384)
        width = kwargs.get("image_width", 384)
        config.encoder.image_size = (height, width)
        config.max_length = kwargs.get("max_length", 20)
        return config

class SinusoidalDeiTEmbeddings(nn.Module):
    """
    Construct the CLS token, distillation token, position and patch embeddings. Optionally, also the mask token.
    """
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())
    
    def __init__(self, embeddings: DeiTEmbeddings, config):
        super().__init__()

        self.cls_token = embeddings.cls_token
        self.distillation_token = embeddings.distillation_token
        self.mask_token = embeddings.mask_token
        self.patch_embeddings = embeddings.patch_embeddings
        if config.encoder.use_learned_position_embeddings:
            self.position_embeddings = embeddings.position_embeddings
        else:
            self.position_embeddings = self.get_embedding(embeddings.patch_embeddings.num_patches + 2, config.encoder.hidden_size).cuda()
        self.dropout = embeddings.dropout

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_length, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class HeightTrOCRProcessor(TrOCRProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        height = kwargs.pop('height', 384)
        width = kwargs.pop('width', 384)
        processor = super().from_pretrained(pretrained_model_name_or_path)
        processor.image_processor.size['height'] = height
        processor.image_processor.size['width'] = width
        return processor

def double_image_height(im1, im2, color=(256, 256, 256)):
    dst = Image.new('RGB', (max(im1.width, im2.width),
                    im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def run_double_height():
    img_height = 384 * 2
    processor = HeightTrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', height=img_height, width=384)
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=True, dec_lpe=True, image_height=img_height, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=True)
    image1 = Image.open("/home/jesse/trocr/IAM/AllImages/a01-000u-00.jpg").convert("RGB")
    image2 = Image.open("/home/jesse/trocr/IAM/AllImages/a01-000u-01.jpg").convert("RGB")
    doubled_height_image = double_image_height(image1, image2)
    pixel_values = processor(doubled_height_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

# The processor is correctly pre-processing the image as the model can read the top and bottom of the doubled image
def test_processor():
    img_height = 384 * 2
    processor = HeightTrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', height=img_height, width=384)
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=True, dec_lpe=True, image_height=384, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=False)
    image1 = Image.open("/home/jesse/trocr/IAM/AllImages/a01-000u-00.jpg").convert("RGB")
    image2 = Image.open("/home/jesse/trocr/IAM/AllImages/a01-000u-01.jpg").convert("RGB")
    doubled_height_image = double_image_height(image1, image2)
    pixel_values = processor(doubled_height_image, return_tensors="pt").pixel_values
    first_phrase, second_phrase = torch.split(pixel_values, 384, 2)
    generated_ids = model.generate(first_phrase)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

if __name__ == "__main__":
    test_processor()