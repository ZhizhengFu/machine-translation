import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch.utils.data import DataLoader, Dataset
import logging
import math
from collections import Counter


TRAIN_FILE = 'cmn-eng-simple/training.txt'
DEV_FILE = 'cmn-eng-simple/validation.txt'
TEST_FILE = 'cmn-eng-simple/testing.txt'
UNK, PAD, BOS, EOS = 0, 1, 2, 3
TYPE = 'test'
GPU = 0
EPOCHS = 100
LAYERS = 6
H_NUM = 8
BATCH_SIZE = 64
D_MODEL = 512
D_FF = 2048
DROPOUT = 0.2
MAX_LENGTH = 80
SAVE_FILE = 'model.pt'
DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename=f'{TYPE}.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def prtlog(msg):
    logging.info(msg)

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [len(seq) for seq in src_batch]
    tgt_lengths = [len(seq) for seq in tgt_batch]
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    src_padded = []
    tgt_padded = []
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded.append(src + [PAD] * (max_src_len - len(src)))
        tgt_padded.append(tgt + [PAD] * (max_tgt_len - len(tgt)))
    return torch.tensor(src_padded), torch.tensor(tgt_padded)

class PrepareData:
    def __init__(self):
        self.train_en, self.train_cn = self.load_data(TRAIN_FILE)
        self.dev_en, self.dev_cn = self.load_data(DEV_FILE)
        self.test_en, self.test_cn = self.load_data(TEST_FILE)
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)
        self.train_en_ids = self.words_to_ids(self.train_en, self.en_word_dict)
        self.train_cn_ids = self.words_to_ids(self.train_cn, self.cn_word_dict)
        self.dev_en_ids = self.words_to_ids(self.dev_en, self.en_word_dict)
        self.dev_cn_ids = self.words_to_ids(self.dev_cn, self.cn_word_dict)
        self.test_en_ids = self.words_to_ids(self.test_en, self.en_word_dict)
        self.test_cn_ids = self.words_to_ids(self.test_cn, self.cn_word_dict)
        self.train_dataset = TranslationDataset(self.train_en_ids, self.train_cn_ids)
        self.dev_dataset = TranslationDataset(self.dev_en_ids, self.dev_cn_ids)
        self.test_dataset = TranslationDataset(self.test_en_ids, self.test_cn_ids)
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, 
                                     shuffle=True, collate_fn=collate_fn)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=BATCH_SIZE, 
                                   shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, 
                                    shuffle=False, collate_fn=collate_fn)

    def load_data(self, path):
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"] + (line[0].lower()).split() + ["EOS"])
                cn.append(["BOS"] + (" ".join([w for w in line[1]])).split() + ["EOS"])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        word_count = Counter()
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1
        
        most_common = word_count.most_common(max_words)
        
        word_dict = {'UNK': UNK, 'PAD': PAD, 'BOS': BOS, 'EOS': EOS}
        for (word, _) in most_common:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
        
        total_words = len(word_dict)
        index_dict = {v: k for k, v in word_dict.items()}
        
        return word_dict, total_words, index_dict

    def words_to_ids(self, sentences, word_dict):
        return [[word_dict.get(word, UNK) for word in sentence] for sentence in sentences]

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt):
        src_mask = self.create_padding_mask(src, PAD)
        tgt_mask = self.create_padding_mask(tgt, PAD)
        tgt_causal_mask = self.create_causal_mask(tgt.size(1))
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            tgt_mask=tgt_causal_mask
        )
        return self.output_projection(output)
    
    def create_padding_mask(self, seq, pad_idx):
        return (seq == pad_idx)
    
    def create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool().to(DEVICE)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        _, _, vocab_size = pred.size()
        pred = pred.reshape(-1, vocab_size)
        target = target.reshape(-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target != self.padding_idx)
        true_dist = true_dist * mask.unsqueeze(1)
        return F.kl_div(F.log_softmax(pred, dim=1), true_dist, reduction='sum')

def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output, tgt_output)
        num_tokens = (tgt_output != PAD).sum().item()
        loss.backward()
        optimizer.step()        
        total_loss += loss.item()
        total_tokens += num_tokens
        
        if batch_idx % 50 == 0:
            prtlog(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()/num_tokens:.4f}")
    
    return total_loss / total_tokens

def evaluate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            num_tokens = (tgt_output != PAD).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
    
    return total_loss / total_tokens

def translate(model, src_sentence, src_dict, tgt_dict, max_len=MAX_LENGTH):
    model.eval()
    src_ids = [src_dict.get(word, UNK) for word in src_sentence]
    src_tensor = torch.tensor([src_ids]).to(DEVICE)
    tgt_ids = [BOS]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids]).to(DEVICE)
            output = model(src_tensor, tgt_tensor)
            next_token = output[0, -1].argmax().item()
            tgt_ids.append(next_token)
            if next_token == EOS:
                break
    
    translation = [tgt_dict.get(idx, 'UNK') for idx in tgt_ids[1:-1]]
    return ' '.join(translation)

def evaluate_bleu(model, data_loader, data):
    model.eval()
    references = []
    translations = []
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(DEVICE)
            batch_size = src.size(0)
            for i in range(batch_size):
                src_sentence = src[i].cpu().tolist()
                src_words = [data.en_index_dict.get(idx, 'UNK') for idx in src_sentence 
                           if idx not in [PAD, BOS, EOS]]
                
                translation = translate(model, src_words, data.en_word_dict, data.cn_index_dict)
                translations.append(translation)
                tgt_sentence = tgt[i].cpu().tolist()
                ref_words = [data.cn_index_dict.get(idx, 'UNK') for idx in tgt_sentence 
                           if idx not in [PAD, BOS, EOS]]
                references.append([' '.join(ref_words)])

    from nltk.translate.bleu_score import corpus_bleu
    prtlog(references)
    prtlog("\n")
    prtlog(translations)
    prtlog("\n")
    bleu1 = corpus_bleu(references, translations, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, translations, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, translations, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, translations, weights=(0.25, 0.25, 0.25, 0.25))
    prtlog(f"BLEU-1: {bleu1:.4f}")
    prtlog(f"BLEU-2: {bleu2:.4f}")
    prtlog(f"BLEU-3: {bleu3:.4f}")
    prtlog(f"BLEU-4: {bleu4:.4f}")
    return bleu4

def main():
    data = PrepareData()
    model = TransformerModel(
        src_vocab_size=data.en_total_words,
        tgt_vocab_size=data.cn_total_words,
        d_model=D_MODEL,
        nhead=H_NUM,
        num_encoder_layers=LAYERS,
        num_decoder_layers=LAYERS,
        dim_feedforward=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)
    
    if TYPE == 'train':
        criterion = LabelSmoothingLoss(data.cn_total_words, PAD, smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        best_loss = float('inf')
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, data.train_loader, criterion, optimizer, epoch)
            prtlog(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            eval_loss = evaluate_epoch(model, data.dev_loader, criterion)
            prtlog(f"Epoch {epoch}, Eval Loss: {eval_loss:.4f}")
            scheduler.step(eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), SAVE_FILE)
                prtlog(f"New best model saved with loss: {best_loss:.4f}")
    
    elif TYPE == 'test':
        model.load_state_dict(torch.load(SAVE_FILE))
        evaluate_bleu(model, data.test_loader, data)

if __name__ == '__main__':
    main()