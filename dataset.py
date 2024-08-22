import json
import os
from typing import Dict
import torch
from torch.utils.data import Dataset
from PIL import Image

from preprocessor import TextPreprocessor, VisionPreprocessor


class CocoDataset(Dataset):
    def __init__(
            self, 
            vision_preprocessor: VisionPreprocessor,
            text_preprocessor: TextPreprocessor,
            data_json_path: str,
            data_folder_path: str
        ):
        self.vision_preprocessor = vision_preprocessor
        self.text_preprocessor = text_preprocessor
        self.data_folder_path = data_folder_path
        self.data_info_list = []

        with open(data_json_path, 'r') as f:
            data_list_from_json = json.load(f)
        self.data_info_list = [item for item in data_list_from_json if 'image' in item.keys() and "coco" in item['image'] ]
        

    def __len__(self):
        return len(self.data_info_list)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        data_info = self.data_info_list[index]

        # Preprocess image
        image_file_name = data_info['image']
        image_file_path = os.path.join(self.data_folder_path, image_file_name)
        image = Image.open(image_file_path).convert('RGB')

        # Preprocess text
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        return data_dict

class GqaDataset(Dataset):
    pass

class OcrVqaDataset(Dataset):
    pass

class TextVqaDataset(Dataset):
    pass

class VgDataset(Dataset):
    pass