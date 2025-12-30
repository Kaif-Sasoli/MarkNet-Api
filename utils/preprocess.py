import torchvision.transforms as transforms
from PIL import Image
# import cv2
import numpy as np

class ResizePad:
    """
    EXACT SAME ResizePad used in training
    """
    def __init__(self, target_h=64, target_w=512, fill=255):
        self.target_h = target_h
        self.target_w = target_w
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        new_w = int(w * (self.target_h / h))

        img = img.resize(
            (min(new_w, self.target_w), self.target_h),
            Image.BILINEAR
        )

        if img.width < self.target_w:
            new_img = Image.new("L", (self.target_w, self.target_h), color=self.fill)
            new_img.paste(img, (0, 0))
            img = new_img
        else:
            img = img.crop((0, 0, self.target_w, self.target_h))

        return img
    
    

# class ResizePadCV:
#     def __init__(self, target_h=64, target_w=512, fill=255):
#         self.target_h = target_h
#         self.target_w = target_w
#         self.fill = fill

#     def __call__(self, img):
#         """
#         img: numpy array (H, W) grayscale
#         returns: normalized float32 array [1, H, W] in [-1,1]
#         """

#         h, w = img.shape[:2]
#         new_w = int(w * (self.target_h / h))

#         # Resize while keeping aspect ratio
#         img = cv2.resize(img, (min(new_w, self.target_w), self.target_h), interpolation=cv2.INTER_AREA)

#         # Pad right if needed
#         if img.shape[1] < self.target_w:
#             pad_width = self.target_w - img.shape[1]
#             pad = np.full((self.target_h, pad_width), self.fill, dtype=np.uint8)
#             img = np.concatenate([img, pad], axis=1)
#         else:
#             img = img[:, :self.target_w]

#         # Normalize to [-1, 1]
#         img = img.astype(np.float32) / 255.0
#         img = (img - 0.5) / 0.5

#         # Add channel dimension
#         return np.expand_dims(img, 0)


class Preprocess:
    """
    EXACT SAME preprocessing as training 
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ResizePad(target_h=64, target_w=512),
            transforms.ToTensor(),              # [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # [-1,1]
        ])

    def __call__(self, img: Image.Image):
        return self.transform(img)