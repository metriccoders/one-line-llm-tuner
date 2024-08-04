from transformers import Trainer, TrainingArguments, AutoModelWithLMHead, AutoTokenizer
from one_line_llm_tuner.reader.file_reader import read_input_file
from one_line_llm_tuner.builder.text_file_builder import build_text_files
from one_line_llm_tuner.dataset.dataset_loader import load_dataset
from sklearn.model_selection import train_test_split

def fine_tune_model(input_file_path, predict_future_text):
  text = read_input_file(input_file_path)
  train, test = train_test_split(text, test_size=0.2)
  build_text_files(train, 'train_dataset.txt')
  build_text_files(test, 'test_dataset.txt')

  tokenizer = AutoTokenizer.from_pretrained('gpt2', truncation=True, padding=True)
  model = AutoModelWithLMHead.from_pretrained('gpt2')

  train_dataset, test_dataset, data_collator = load_dataset('train_dataset.txt', 'test_dataset.txt', tokenizer)

  training_args = TrainingArguments(output_dir='./results', num_train_epochs=2, logging_steps=100, save_steps=100, per_device_train_batch_size=64, per_device_eval_batch_size=64)

  trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=test_dataset)
  trainer.train()
  trainer.save_model()

  input_ids = tokenizer.encode(predict_future_text, return_tensors='pt').to('cuda')

  output=model.generate(input_ids, max_length=500, num_return_sequences=1)

  return tokenizer.decode(output[0], skip_special_tokens=True)



