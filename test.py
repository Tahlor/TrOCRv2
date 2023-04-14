from src.TestHelpers.TestRunner import run
from src.PositionEmbeddings.SinusoidalDeitEmbedding import SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig
from transformers import TrOCRProcessor, BertTokenizer
import sys

def main():
    num_workers = 8
    batch_size = 18
    image_directory = '/home/jesse/TrOCR/'
    save_directory = '/home/jesse/TrOCR/Tests/SinusoidalSingle/SinusoidalSingleLR/Results/Model'

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    processor.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=False, dec_lpe=False, image_height=384, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=False)
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    lr = 5e-5 / 10

    run(processor, model, batch_size, num_workers, image_directory, 50, lr, save_directory)

main()