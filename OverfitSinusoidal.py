from src.TestHelpers.TestRunner import run
from src.PositionEmbeddings.SinusoidalDeitEmbedding import SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig
from transformers import TrOCRProcessor
import sys

def main():
    num_workers = 9
    batch_size = 16
    image_directory = '/home/jesse/TrOCR/'

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=False, dec_lpe=False, image_height=384, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=False)
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    lr = 5e-5

    run(processor, model, lr, batch_size=batch_size, num_workers=num_workers, root_directory=image_directory, warmup=False, num_epochs=201, print_train_cer_every=10, print_eval_cer_every=5, print_num_samples=10)

main()