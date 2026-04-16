
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SemanticRefiner(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 256,
                 max_len: int = 128,
                 freeze_encoder: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        try:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.encoder   = BertModel.from_pretrained("bert-base-uncased")
            self.bert_dim  = 768
            self._has_bert = True
        except ImportError:
            # Fallback: simple learned embedding if transformers not installed
            self._has_bert = False
            self.bert_dim  = 256
            self.embed = nn.Embedding(vocab_size + 1, self.bert_dim,
                                      padding_idx=0)

        if freeze_encoder and self._has_bert:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Trainable decoder head 
        # Projects BERT hidden states → corrected gloss logits
        self.decoder_head = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    
    def forward(self, gloss_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self._has_bert:
            out = self.encoder(
                input_ids=gloss_ids,
                attention_mask=attention_mask,
            )
            hidden = out.last_hidden_state   # (B, L, 768)
        else:
            hidden = self.embed(gloss_ids)   # (B, L, 256)

        return self.decoder_head(hidden)     # (B, L, vocab_size)


    @torch.no_grad()
    def refine(self, gloss_sequence: List[str],
               beam_width: int = 4) -> List[str]:
        
        if not self._has_bert:
            return gloss_sequence

        text = " ".join(gloss_sequence)
        enc  = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        ids  = enc["input_ids"].to(next(self.parameters()).device)
        mask = enc["attention_mask"].to(ids.device)

        logits = self.forward(ids, mask)          # (1, L, vocab_size)
        pred   = logits.argmax(dim=-1).squeeze(0) # (L,)

        tokens = self.tokenizer.convert_ids_to_tokens(pred.tolist())
        # Strip special tokens and return
        tokens = [t for t in tokens
                  if t not in ("[CLS]", "[SEP]", "[PAD]", "[UNK]")]
        return tokens


class GlossVocab:
    BLANK = "<blank>"
    PAD   = "<pad>"
    UNK   = "<unk>"

    def __init__(self, glosses: Optional[List[str]] = None):
        specials = [self.BLANK, self.PAD, self.UNK]
        self._idx2g: List[str] = specials[:]
        self._g2idx = {g: i for i, g in enumerate(specials)}
        if glosses:
            for g in glosses:
                self.add(g)

    def add(self, gloss: str):
        if gloss not in self._g2idx:
            self._g2idx[gloss] = len(self._idx2g)
            self._idx2g.append(gloss)

    def __len__(self):
        return len(self._idx2g)

    @property
    def blank_id(self):
        return self._g2idx[self.BLANK]

    def encode(self, glosses: List[str]) -> List[int]:
        return [self._g2idx.get(g, self._g2idx[self.UNK]) for g in glosses]

    def decode(self, ids: List[int]) -> List[str]:
        return [self._idx2g[i] for i in ids
                if i < len(self._idx2g) and self._idx2g[i] != self.BLANK]

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self._idx2g, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GlossVocab":
        import json
        with open(path) as f:
            glosses = json.load(f)
        vocab = cls()
        for g in glosses:
            vocab.add(g)
        return vocab
