from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModelWithLMHead, AutoTokenizer
import re
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split