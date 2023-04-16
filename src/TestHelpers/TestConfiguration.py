import yaml
from yaml.loader import SafeLoader

class TestConfiguration:
    def __init__(self, **kwargs):
        self.batch_size = 23
        self.num_workers = 9
        self.root_directory = '/home/jclar234/TrOCR'
        self.num_epochs = 50
        self.save_directory = None
        self.save_every = 5
        self.print_train_cer_every = 5
        self.print_eval_cer_every = 5
        self.print_loss_every = 1
        self.print_num_samples = 1
        self.train = True
        self.unfreeze = False
        self.use_double = False
        self.num_images = None
        self.constant_warmup = False
        self.linear_schedule_with_warmup = False
        self.processor_pretrained_path = 'microsoft/trocr-small-handwritten'
        self.model_pretrained_path = 'microsoft/trocr-small-handwritten'
        self.image_height = 384
        self.image_width = 384
        self.max_tokens = 50
        self.tokenizer_type = None
        self.use_learned_position_embeddings = True
        self.lr = 5e-5
        self.train_only_embeddings = False

        self.__dict__.update(kwargs)

    @staticmethod
    def from_file(config_file_path):
        data_dictionary = None
        with open(config_file_path, 'r') as config_file:
            data_dictionary = yaml.load(config_file, Loader=SafeLoader)
        return TestConfiguration(**data_dictionary)
    
    def write_yaml(self, out_path):
        with open(out_path, 'w') as out_config:
            yaml.dump(vars(self), out_config)