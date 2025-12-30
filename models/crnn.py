import torch
import torch.nn as nn
from PIL import Image
from utils.preprocess import Preprocess
import torch.nn.functional as F
from utils.utils import (
    beam_decoder,
    decode_torchaudio_results,
    clean_field
)



# CRNN Model
class CRNNModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNNModel, self).__init__()

        # Combine Conv layers
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=512 * 4,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # Fully connected (2 directions → hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        b, c, h, w = x.size()

        # Reshape for LSTM (sequence length = width)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        # LSTM
        x, _ = self.lstm(x)

        # Final Layer
        x = self.fc(x)

        return x




# Predictor Class
class CRNNPredictor:
    def __init__(self, model, vocab, device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.blank = 0

        self.preprocess = Preprocess()
        

    def predict(self, pil_image, cls=None):
        img = pil_image.convert("L")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)           # [1, T, C]
            log_probs = F.log_softmax(logits, dim=2)  

            results = beam_decoder(log_probs.cpu())

            pred_text = decode_torchaudio_results(results)
            
        # Clean the text based on class
        cleaned_text = clean_field(pred_text, cls) if cls is not None else pred_text
        
        # Return "N/A" if empty
        if not cleaned_text or cleaned_text.strip() == "":
            return "N/A"
        
        return cleaned_text
            