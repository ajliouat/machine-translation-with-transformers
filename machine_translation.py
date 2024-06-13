import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from sacrebleu import corpus_bleu

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_length=128):
        self.src_sentences = self.read_file(src_file)
        self.tgt_sentences = self.read_file(tgt_file)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        src_encoded = self.src_tokenizer.encode_plus(src_sentence, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        tgt_encoded = self.tgt_tokenizer.encode_plus(tgt_sentence, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'src_input_ids': src_encoded['input_ids'].squeeze(),
            'src_attention_mask': src_encoded['attention_mask'].squeeze(),
            'tgt_input_ids': tgt_encoded['input_ids'].squeeze(),
            'tgt_attention_mask': tgt_encoded['attention_mask'].squeeze()
        }

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_embed = self.positional_encoding(self.encoder_embedding(src))
        tgt_embed = self.positional_encoding(self.decoder_embedding(tgt))
        outs = self.transformer(src_embed, tgt_embed, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.fc(outs)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(device)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train(model, dataloader, optimizer, scheduler, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            src = batch['src_input_ids'].to(device)
            tgt = batch['tgt_input_ids'].to(device)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, pad_idx)

            optimizer.zero_grad()
            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    references = []
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src_input_ids'].to(device)
            tgt = batch['tgt_input_ids'].to(device)
            src_mask, _, src_padding_mask, _ = create_mask(src, src, pad_idx)
            output = model(src, src, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
            output = output.argmax(dim=-1)
            predictions.extend(tgt_tokenizer.batch_decode(output, skip_special_tokens=True))
            references.extend(tgt_tokenizer.batch_decode(tgt, skip_special_tokens=True))
    bleu_score = corpus_bleu(predictions, [references]).score
    print(f"BLEU score: {bleu_score:.2f}")

# Set up the dataset, tokenizers, and data loaders
src_file = 'french_sentences.txt'
tgt_file = 'english_sentences.txt'
src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_dataset = TranslationDataset(src_file, tgt_file, src_tokenizer, tgt_tokenizer)
val_dataset = TranslationDataset(src_file, tgt_file, src_tokenizer, tgt_tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Set up the model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = src_tokenizer.vocab_size
tgt_vocab_size = tgt_tokenizer.vocab_size
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
dropout = 0.1
model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
num_training_steps = len(train_dataloader) * epochs
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

# Train and evaluate the model
epochs = 10
pad_idx = src_tokenizer.pad_token_id
train(model, train_dataloader, optimizer, scheduler, criterion, device, epochs)
evaluate(model, val_dataloader, device)
