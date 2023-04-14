from src.TestHelpers.TestRunner import run
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys

def main():
    num_workers = sys.argv[1]
    batch_size = sys.argv[2]
    root_directory = sys.argv[3]
    save_directory = sys.argv[4]

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')
    model.config.decoder_start_token_id = 2
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    lr = 5e-5 / (2048 / batch_size)

    run(processor, model, lr, batch_size, num_workers, root_directory=root_directory, save_directory=save_directory)

main()