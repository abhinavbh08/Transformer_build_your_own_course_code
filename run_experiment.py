from torch import optim
from read_data import load_data, read_val_data
from models import TransformerEncoderDecoder, TransformerEncoder, TransformerDecoder
from loss import MaskedCELoss
import torch
import torch.nn as nn
from read_data import truncate_and_pad
from nltk.translate.bleu_score import corpus_bleu
import nltk
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_sentence(model, sentence, src_vocab, tgt_vocab, max_len, device):

    model.eval()
    # get the src tokens in indexed form.
    src_tokens = src_vocab[nltk.tokenize.word_tokenize(sentence.lower())] + [src_vocab["<eos>"]]
    # Get the lengths of the input sentence
    x_len = torch.tensor([len(src_tokens)], device=device)
    # truncate and pad the input sentence
    src_tokens = truncate_and_pad(src_tokens, max_len, src_vocab["<pad>"])
    # Creating a batch from a single input sentence.
    x = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_output = model.encoder(x, x_len)    # (B, max_len, hidden_dim)
    state = model.decoder.init_state(enc_output, x_len)
    y = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    # Generate the output tokens one by one
    for i in range(max_len):
        output, state = model.decoder(y, state) # (B, 1, tgt_vocab_size)
        y = output.argmax(dim=2)
        # Get the predicted token label.
        y_pred = y.squeeze(dim=0).type(torch.int32).item()
        # quit if end of sentence is reached.
        if y_pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(y_pred)

    # Converting the output generated sequence to words using target language vocabulary.
    return " ".join([tgt_vocab.idx2word[item] for item in output_seq])

def test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed):

    predictions = []
    for sentence in tqdm(sentences_preprocessed):
        sentence_predicted = predict_sentence(model, sentence, src_vocab, tgt_vocab, len_sequence, device)
        predictions.append(sentence_predicted)

    references = [[nltk.tokenize.word_tokenize(sent.lower())] for sent in true_trans_preprocessed]
    candidates = [nltk.tokenize.word_tokenize(sent.lower()) for sent in predictions]
    score = corpus_bleu(references, candidates)
    print(score)

def train_model(model, data_loader, learning_rate, n_epochs, tgt_vocab, src_vocab, device):

    # Masked Cross Entropy loss function.
    loss_function = MaskedCELoss()

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    # Do Xavier initialisatation of the weights.
    model.apply(initialize_weights)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Putting the model to train mode.
    model.train()

    sentences_preprocessed, true_trans_preprocessed = read_val_data()
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            x, x_len, y, y_len = [item.to(device) for item in batch]
            bos_token = torch.tensor([tgt_vocab["<bos>"]] * y.shape[0], device=device).reshape(-1, 1)
            # Append beginning of sentence (BOS) to the input to the decoder so that input to the decoder is shifted by one to the right.
            decoder_input = torch.cat([bos_token, y[:, :-1]], 1)
            output_model, state = model(x, decoder_input, x_len)
            # Passing the output of the model to the loss function.
            l = loss_function(output_model, y, y_len)
            # Backpropagate the loss
            l.sum().backward() 
            # Do gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                running_loss += l.sum().item()

        # Save model every 19th epoch.
        if epoch % 20 == 19:
            PATH = "model_att.pt"
            torch.save(model.state_dict(), PATH)

        test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed)
        print(f"Epoch_Loss, {epoch}, {running_loss / len(data_loader.dataset)}")

batch_size = 64
len_sequence = 15
lr = 0.0008
n_epochs = 60
print(n_epochs, lr, len_sequence)

data_iter, src_vocab, tgt_vocab = load_data(batch_size, len_sequence)
print(len(src_vocab))
print(len(tgt_vocab))
encoder = TransformerEncoder(
    query=128, key=128, value=128, hidden_size=128, num_head=4, dropout=0.1, lnorm_size=[128], ffn_input=128, ffn_hidden=256, vocab_size=len(src_vocab), num_layers = 2
)
decoder = TransformerDecoder(
    query=128, key=128, value=128, hidden_size=128, num_head=4, dropout=0.1, lnorm_size=[128], ffn_input=128, ffn_hidden=256, vocab_size=len(tgt_vocab), num_layers = 2
)
model = TransformerEncoderDecoder(encoder, decoder)
train_model(model, data_iter, lr, n_epochs, tgt_vocab, src_vocab, device)
PATH = "model_att.pt"
torch.save(model.state_dict(), PATH)


# model.load_state_dict(torch.load(PATH, map_location=device))
# sentences = ["This is a nice place."]
# sentences_preprocessed = [sentence for sentence in sentences]
# predictions = []
# for sentence in sentences_preprocessed:
#     sentence_predicted = predict_sentence(model, sentence, src_vocab, tgt_vocab, len_sequence, device)
#     print(sentence_predicted)
#     predictions.append(sentence_predicted)
