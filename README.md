# Ngram: Character-level Ngram Neural Network Model

Ngram is a character-level Ngram-based neural network model that generates names using a probabilistic model. The implementation is designed to train and generate sequences of characters, relying on an N-gram approach combined with neural network techniques.

This project consists of two main scripts:

- **`ngram_train.py`**: A script to train the model on a given dataset.
- **`ngram_generate.py`**: A script to generate text from the trained model.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Generating Text](#generating-text)
- [License](#license)

## Installation

To use the Ngram model, you will need Python 3.x and several dependencies. It is recommended to use a virtual environment.

1. Clone the repository:
   ```bash
   git clone https://github.com/cancer32/NGram.git
   cd NGram
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, run the `ngram_train.py` script with the required arguments.

```bash
python ngram_train.py --dataset_path 'dataset/names.txt' --checkpoint_path 'weights/names.pt' [--n_gram <n>] [--seed <random_seed>] [--epochs <epochs>] [--lr <learning_rate>]
```

**Arguments:**
- `--dataset_path`: The path to the text dataset for training (required).
- `--checkpoint_path`: The directory to save the trained model weights (required).
- `--n_gram`: The N-gram size (default is 4).
- `--seed`: Random seed for reproducibility (default is random).
- `--epochs`: The number of epochs for training (default is 100).
- `--lr`: The learning rate for training (default is 10).


### Generating Text

Once the model is trained, you can generate text by running the `ngram_generate.py` script.

```bash
python ngram_generate.py --checkpoint_path 'weights/names.pt' [--seed <random_seed>] [--count <num_words>] [--start <start_string>]
```

**Arguments:**
- `--checkpoint_path`: The path to the saved model weights (required).
- `--seed`: Random seed for text generation (default is random).
- `--count`: The number of words (or characters) to generate (default is 10).
- `--start`: The starting string from which to generate text (optional).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute or raise issues for improvements!