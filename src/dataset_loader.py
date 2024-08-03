from transformers import TextDataset, DataCollatorForLanguageModeling

def load_dataset(train_path,test_path,tokenizer):
  train_dataset = TextDataset(tokenizer=tokenizer,file_path=train_path,block_size=64)
  test_dataset = TextDataset(tokenizer=tokenizer,file_path=test_path, block_size=64)
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
  return train_dataset,test_dataset,data_collator