


import torch
from dataset import CocoDataset
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava import conversation as conversation_lib
import transformers

from llava.train.train import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedDataset
from preprocessor import TextPreprocessor, VisionPreprocessor


if __name__ == "__main__":


    # Set default_conversation
    default_conversation = conversation_lib.conv_templates["v1"]

    # Set is_multimodal
    is_multimodal = True

    # Set mm_use_im_start_end
    mm_use_im_start_end = False

    # Set tokenizer
    tokenizer =  transformers.AutoTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    image_processor =  CLIPVisionTower("openai/clip-vit-large-patch14-336",-2, "patch" ).image_processor

    # Set dataset
    data_args = DataArguments(
        data_path="/home/hyeonan/LLaVA/playground/data/llava_v1_5_mix665k.json",
        lazy_preprocess=True,
        image_folder="/home/hyeonan/LLaVA/playground/data",
        image_aspect_ratio="pad",
        is_multimodal=True,
        image_processor = image_processor,
        mm_use_im_start_end = mm_use_im_start_end
    )
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path="/home/hyeonan/LLaVA/playground/data/llava_v1_5_mix665k.json",
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_batch_size = 1
    dataloader_params = {
            "batch_size": train_batch_size,
            "collate_fn": data_collator,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
    batch = next(iter(train_loader))
    print(torch.all(batch["labels"] == -100))
    print(batch["labels"].shape)