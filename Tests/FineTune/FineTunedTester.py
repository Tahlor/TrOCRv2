from TestHelpers.TestRunner import run
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys

def main():
    num_workers = sys.argv[1]
    batch_size = sys.argv[2]
    image_directory = sys.argv[3]
    pretrained_path = sys.argv[4]

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_path)

    run(processor, model, batch_size, num_workers, image_directory, 1, train=False, print_train_cer_every=1, print_eval_cer_every=1, print_num_samples=5)

main()