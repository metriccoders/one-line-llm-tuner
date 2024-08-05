from transformers import TextDataset, DataCollatorForLanguageModeling


def load_dataset(train_path, test_path, tokenizer):
      """
      Load the training and testing dataset along with data collator
      :param train_path:
      :param test_path:
      :param tokenizer:
      :return: training dataset object, testing dataset object, data collator object
      """
      train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=64)
      test_dataset = TextDataset(tokenizer=tokenizer, file_path=test_path, block_size=64)
      data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
      return train_dataset, test_dataset, data_collator
