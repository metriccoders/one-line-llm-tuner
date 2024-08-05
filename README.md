# one-line-llm-tuner

Fine-tune a Large Language Model (LLM) in a single line.

## Overview

`one-line-llm-tuner` is a Python package designed to simplify the process of fine-tuning large language models (LLMs) like GPT-3. With just one line of code, you can adapt a pre-trained model to your specific dataset.

## Features

- **Simple API**: Fine-tune models with minimal code.
- **Supports Popular LLMs**: Works with models from the `transformers` library, including GPT, BERT, and more.
- **Customizable**: Advanced users can customize the fine-tuning process with additional parameters.

## Installation

You can install `one-line-llm-tuner` using pip:

```bash
pip install one-line-llm-tuner
```

## Usage

The PyPI package can be used in the following way after installation.

```bash
from one_line_llm_tuner.tuner import llm_tuner

fine_tune_obj = llm_tuner.FineTuneModel()

fine_tune_obj.fine_tune_model(input_file_path="train.txt")

fine_tune_obj.predict_text("Elon musk founded Spacex in ")
```