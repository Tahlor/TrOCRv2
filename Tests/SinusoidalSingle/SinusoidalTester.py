from src.TestHelpers.TestRunner import run
from transformers import TrOCRProcessor
from src.PositionEmbeddings.SinusoidalDeitEmbedding import SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig
import sys

def main():
    num_workers = sys.argv[1]
    batch_size = sys.argv[2]
    image_directory = sys.argv[3]
    pretrained_path = sys.argv[4]

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=False, dec_lpe=False, image_height=384, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained(pretrained_path, config=config, ignore_mismatched_sizes=False)

    run(processor, model, batch_size, num_workers, image_directory, 1, train=False, print_train_cer_every=1, print_eval_cer_every=1, print_num_samples=5)

main()