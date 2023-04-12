from TestHelpers.TestRunner import run
from src.PositionEmbeddings.SinusoidalDeitEmbedding import HeightTrOCRProcessor, SinusoidalVisionEncoderDecoder, SinusoidalVisionEncoderDecoderConfig
import sys

def disable_grad(model):
    for name, param in model.named_parameters():
        if name != "encoder.embeddings.position_embeddings" and name != "decoder.model.decoder.embed_positions.weight":
            param.requires_grad = False

def main():
    num_workers = sys.argv[1]
    batch_size = sys.argv[2]
    image_directory = sys.argv[3]
    save_directory = sys.argv[4]

    new_height = 384 * 2
    processor = HeightTrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', height=new_height, width=384)
    config = SinusoidalVisionEncoderDecoderConfig.from_pretrained('microsoft/trocr-small-handwritten', enc_lpe=True, dec_lpe=True, image_height=new_height, image_width=384, max_length=50)
    model = SinusoidalVisionEncoderDecoder.from_pretrained('microsoft/trocr-small-handwritten', config=config, ignore_mismatched_sizes=False)
    
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = 2
    model.config.sep_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    disable_grad(model)
    lr = 5e-5

    run(processor, model, batch_size, num_workers, image_directory, 50, lr, save_directory, use_double = True)

main()