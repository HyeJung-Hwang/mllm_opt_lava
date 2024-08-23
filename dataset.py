import json
import os
from typing import Dict
import torch
from torch.utils.data import Dataset
from PIL import Image

from preprocessor import TextPreprocessor, VisionPreprocessor
from llava import conversation as conversation_lib


class CocoDataset(Dataset):
    def __init__(
            self, 
            vision_tower,
            vision_preprocessor: VisionPreprocessor,
            text_preprocessor: TextPreprocessor,
            data_json_path: str,
            data_folder_path: str,
            default_conversation: conversation_lib.Conversation,
            is_multimodal: bool,
            mm_use_im_start_end: bool,
            tokenizer,
        ):
        self.vision_tower = vision_tower
        self.vision_preprocessor = vision_preprocessor
        self.text_preprocessor = text_preprocessor
        self.data_folder_path = data_folder_path
        self.data_info_list = []
        self.default_conversation = default_conversation
        self.is_multimodal = is_multimodal
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer

        with open(data_json_path, 'r') as f:
            data_list_from_json = json.load(f)
        self.data_info_list = [item for item in data_list_from_json if 'image' in item.keys() and "coco" in item['image'] ]
        

    def __len__(self):
        return len(self.data_info_list)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data_info_list:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data_info_list:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        data_info = self.data_info_list[index]

        # Check if data has image data
        has_image = 'image' in data_info.keys()

        # Run Preprocess 
        data_dict = {}
        if has_image:
            # Run vision preprocess
            image_file_name = data_info['image']
            image_file_path = os.path.join(self.data_folder_path, image_file_name)
            image = Image.open(image_file_path).convert('RGB')
            data_dict['image'] = self.vision_preprocessor(image)
            # Run text preprocess 
            text = data_info['conversations']
            print(self.text_preprocessor(text,has_image))
            # data_dict['input_ids'] = input_ids
            # data_dict['labels'] = labels
        else:
            # Make fake image data
            crop_size = self.vision_tower.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # Run text preprocess
            text = data_info['conversations']
            input_ids, labels = self.text_preprocessor(text, has_image)
            data_dict['input_ids'] = input_ids
            data_dict['labels'] = labels

        return data_dict

class GqaDataset(Dataset):
    pass

class OcrVqaDataset(Dataset):
    pass

class TextVqaDataset(Dataset):
    pass

class VgDataset(Dataset):
    pass