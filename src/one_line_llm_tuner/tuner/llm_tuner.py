from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from one_line_llm_tuner.reader.file_reader import read_input_file
from one_line_llm_tuner.builder.text_file_builder import build_text_files
from one_line_llm_tuner.dataset.dataset_loader import load_dataset
from sklearn.model_selection import train_test_split


class FineTuneModel:
    """
    A class to fine tune a Large Language Model

    Attributes
    ----------
    model_name : str
        The name of the base model

    test_size : float
        The size of the test dataset in decimals

    training_dataset_filename : str
        The name of the training dataset file

    testing_dataset_filename : str
        The name of the testing dataset file

    tokenizer_truncate : bool
        To truncate the tokens or not

    tokenizer_padding : bool
        To pad the tokens or not

    output_dir : str
        The default directory of the output

    num_train_epochs : int
        The default number of training epochs

    logging_steps : int
        The number steps before logging the evaluation

    save_steps : int
        The number of steps before saving the evaluation

    per_device_train_batch_size : int
        The batch size of training tokens

    per_device_eval_batch_size=64 : int
        The batch size of evaluation tokens

    max_output_length : int
        The maximum number of output tokens

    num_return_sequences : int
        The number of return sequences

    skip_special_tokens : bool
        To skip special tokens or not


    Methods
    -------
    get_tokenizer():
        Returns the tokenizer

    get_init_model():
        Returns the model

    fine_tune_model():
        Fine-tune the LLM

    predict_text():
        Perform text prediction

    """
    def __init__(self,
                 model_name="gpt2",
                 test_size=0.2,
                 training_dataset_filename="train_dataset.txt",
                 testing_dataset_filename="test_dataset.txt",
                 tokenizer_truncate=True,
                 tokenizer_padding=True,
                 output_dir="./results",
                 num_train_epochs=2,
                 logging_steps=100,
                 save_steps=100,
                 per_device_train_batch_size=64,
                 per_device_eval_batch_size=64,
                 max_output_length=500,
                 num_return_sequences=1,
                 skip_special_tokens=True,
                 ):
        self.model_name = model_name
        self.test_size = test_size
        self.training_dataset_filename = training_dataset_filename
        self.testing_dataset_filename = testing_dataset_filename
        self.tokenizer_truncate = tokenizer_truncate
        self.tokenizer_padding = tokenizer_padding

        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, truncation=self.tokenizer_truncate,
                                                       padding=self.tokenizer_padding)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.training_args = TrainingArguments(output_dir=self.output_dir,
                                               num_train_epochs=self.num_train_epochs,
                                               logging_steps=self.logging_steps,
                                               save_steps=self.save_steps,
                                               per_device_train_batch_size=self.per_device_train_batch_size,
                                               per_device_eval_batch_size=self.per_device_eval_batch_size)

        self.max_output_length = max_output_length
        self.num_return_sequences = num_return_sequences
        self.skip_special_tokens = skip_special_tokens

        self.trainer = Trainer(model=self.model, args=self.training_args)

    def get_tokenizer(self):
        """
        Returns the tokenizer object
        :return: Tokenizer object
        """
        return self.tokenizer

    def get_init_model(self):
        """
        Returns the model object
        :return: Model object
        """
        return self.model

    def fine_tune_model(self, input_file_path):
        """
        Fine tune the Large Language Model
        :param input_file_path:
        :return: None
        """
        try:
            text = read_input_file(input_file_path)
            train, test = train_test_split(text, test_size=0.2)
            build_text_files(train, self.training_dataset_filename)
            build_text_files(test, self.testing_dataset_filename)

            train_dataset, test_dataset, data_collator = load_dataset('train_dataset.txt', 'test_dataset.txt',
                                                                      self.tokenizer)

            self.trainer = Trainer(model=self.model, args=self.training_args, data_collator=data_collator,
                                   train_dataset=train_dataset, eval_dataset=test_dataset)

            self.trainer.train()

            self.trainer.save_model()

        except Exception as e:
            print(f"Caught an exception while fine tuning the model: {e}")

    def predict_text(self, input_text):
        """
        Prediction of future text
        :param input_text:
        :return: Output text
        """
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to('cuda')
            output = self.model.generate(input_ids, max_length=self.max_output_length,
                                         num_return_sequences=self.num_return_sequences)
            return self.tokenizer.decode(output[0],
                                         skip_special_tokens=self.skip_special_tokens)
        except Exception as e:
            print(f"Caught an exception while predicting text: {e}")
