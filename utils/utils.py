# Utils
import re
from datetime import datetime
from torchaudio.models.decoder import ctc_decoder
import torch.nn.functional as F
import torch
import torch.nn as nn


DEFAULT_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-%:|, "

# Month mapping
MONTH_MAP = {
    "JAN": "Jan", "FEB": "Feb", "MAR": "Mar", "APR": "Apr",
    "MAY": "May", "JUN": "Jun", "JUL": "Jul", "AUG": "Aug",
    "SEP": "Sep", "OCT": "Oct", "NOV": "Nov", "DEC": "Dec"
}


# 
char_to_idx = {c: i + 1 for i, c in enumerate(DEFAULT_CHARS)}  
idx_to_char = {i + 1: c for i, c in enumerate(DEFAULT_CHARS)}  


# Clean Field based on class
def clean_field(text, cls):
    text = text.strip()

    # Normalize OCR noise
    text = text.replace('|', ' ')
    text = text.replace('%', ':')
    text = re.sub(r'\s+', ' ', text)

    # NAME 
    if cls == 2:
        name = re.sub(r'(?i)NAME\s*:', '', text).strip()
        return f"NAME: {name}"

    # PERCENTAGE 
    elif cls == 3:
        value = re.sub(r'(?i)PERCENTAGE\s*:', '', text).strip()

        # Fix separators
        value = value.replace('-', '.').replace(':', '')
        value = re.findall(r'\d+\.?\d*', value)

        if value:
            value = float(value[0])
            return f"PERCENTAGE: {value:.2f}%"

        return "PERCENTAGE:"

    # DATE 
    elif cls == 0:
        date_part = re.sub(r'(?i)DATE OF ISSUANCE\s*:', '', text).strip()

        # Convert 24.OCT.2022 → 24-Oct-2022
        def fix_date(m):
            day, mon, year = m.groups()
            mon = MONTH_MAP.get(mon.upper(), mon)
            return f"{int(day):02d}-{mon}-{year}"

        date_part = re.sub(
            r'(\d{1,2})[.\- ]([A-Z]{3})[.\- ](\d{4})',
            fix_date,
            date_part
        )

        return f"DATE OF ISSUANCE: {date_part}"

    # GROUP
    elif cls == 1:
        group = re.sub(r'(?i)GROUP\s*:', '', text).strip()
        return f"GROUP: {group}"

    return text


# IMPORTANT: blank MUST be index 0
labels = ["<blank>"] + list(DEFAULT_CHARS)

beam_decoder = ctc_decoder(
    lexicon=None,
    tokens=labels,
    blank_token="<blank>",
    beam_size=20,   # you can try 5 or 10
)  

# Decode torchaudio ctc_decoder results
def decode_torchaudio_results(results):
    """
    results: output of torchaudio ctc_decoder
    """
    # we have batch size = 1
    best = results[0][0]        
    tokens = best.tokens

    text = "".join(
        DEFAULT_CHARS[t - 1] for t in tokens if t > 0
    )
    return text



# Encode Text
def encode_text(s):
    return [char_to_idx[c] for c in s.upper() if c in char_to_idx]

# Decode Text
def decode_indices(seq):
    out, prev = [], None
    for i in seq:
        if i != prev and i != 0:
            out.append(idx_to_char[i])
        prev = i
    return "".join(out)

 
 

# Prediction Decoding (CTC)
def decode_predictions(log_probs_batch, blank=0):
    texts = []

    for log_probs in log_probs_batch:      # [T, C]
        preds = torch.argmax(log_probs, dim=1)

        prev = blank
        decoded = []

        for p in preds:
            p = p.item()
            if p != blank and p != prev:
                decoded.append(p)
            prev = p

        texts.append(decode_indices(decoded))
    print(texts)
    return texts
