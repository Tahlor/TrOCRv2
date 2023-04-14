from transformers import TrOCRProcessor

def main():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    processor

main()