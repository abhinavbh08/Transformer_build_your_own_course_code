import torch
from vocab import Vocab
from torch.utils import data
import nltk


def read_data():
    """Reads the training dataset"""

    # Data paths if training models on kaggle kernels for better gpus
    data_path_en_kaggle = "/kaggle/input/deentxt/train.en"
    data_path_de_kaggle = "/kaggle/input/deentxt/train.de"

    # data path if reading from local directory
    data_path_en = "data/de-en_deduplicated/train.en"
    data_path_de = "data/de-en_deduplicated/train.de"

    with open(data_path_en_kaggle, "r") as file:
        data_en = file.read().split("\n")[:-1]

    with open(data_path_de_kaggle, "r") as file:
        data_de = file.read().split("\n")[:-1]        

    return data_en, data_de


def read_val_data():
    """Reads the validation data and returns the list of source and target sentences."""

    data_path_en_kaggle = "/kaggle/input/deentxt/val.en"
    data_path_de_kaggle = "/kaggle/input/deentxt/val.de"

    data_path_en = "data/de-en_deduplicated/val.en" 
    data_path_de = "data/de-en_deduplicated/val.de" 

    with open(data_path_en_kaggle, "r") as file:
        data_en = file.read().split("\n")[:-1]
    with open(data_path_de_kaggle, "r") as file:
        data_de = file.read().split("\n")[:-1]        

    source, target = data_en, data_de
    return source, target


def tokenize_data(text):
    """Tokenize the data using NLTK word tokenizer."""

    source, target = [], []
    if isinstance(text, tuple):
        for sent in text[0]:
            source.append(nltk.tokenize.word_tokenize(sent.lower()))
        for sent in text[1]:
            target.append(nltk.tokenize.word_tokenize(sent.lower()))
            
    # Return lists of tokenized src and target sentences.
    return source, target


def truncate_and_pad(line, max_len, padding_token):
    """Truncate the sentence if it is longer than max_len, also pad it to max_len if it is smaller than max_len"""

    # Truncate the sentence.
    if len(line) > max_len:
        return line[:max_len]
    
    # pad the sentence if it is less than max_len.
    return line + [padding_token] * (max_len - len(line))


def convert_to_indices(lines, vocab, max_len):
    """Convert the data to indices and also get the original lengths of the sentences before padding."""

    # Get the indices from the vocab for each of the sentence.
    lines = [vocab[l] for l in lines]

    # Append end of sentence token to the sentence.
    lines = [l + [vocab["<eos>"]] for l in lines]
    arr = torch.tensor([truncate_and_pad(l, max_len, vocab["<pad>"]) for l in lines])

    # Caclulate the actual lengths of the sentence before padding.
    valid_len = (arr != vocab["<pad>"]).type(torch.int32).sum(1)
    return arr, valid_len


def load_data(batch_size, max_len):
    """Load the data into a dataloader."""

    preprocessed_text = read_data()
    source, target = tokenize_data(preprocessed_text)

    # Creating the source vocab
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # Creating the target vocab
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # Converting the data to indices.
    src_array, src_valid_len = convert_to_indices(source, src_vocab, max_len)
    tgt_array, tgt_valid_len = convert_to_indices(target, tgt_vocab, max_len)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    dataset = data.TensorDataset(*data_arrays)

    # Create the dataloader from the dataset.
    itrtr = data.DataLoader(dataset, batch_size, shuffle=True)
    return itrtr, src_vocab, tgt_vocab