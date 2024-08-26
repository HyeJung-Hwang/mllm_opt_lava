


from typing import Dict
import torch
from dataset import CocoDataset
from llava.model.langauge_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava import conversation as conversation_lib
import transformers

from llava.train.train import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedDataset, ModelArguments, TrainingArguments, get_model
from preprocessor import TextPreprocessor, VisionPreprocessor

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

if __name__ == "__main__":


    # Set dataset
    data_args = DataArguments(
        data_path="/home/hyeonan/LLaVA/playground/data/llava_v1_5_mix665k.json",
        lazy_preprocess=True,
        image_folder="/home/hyeonan/LLaVA/playground/data",
        image_aspect_ratio="pad",
    )

    # Set model
    model_args = ModelArguments(
        model_name_or_path="lmsys/vicuna-7b-v1.5",
        version="v1",
        vision_tower="openai/clip-vit-large-patch14-336",
        pretrain_mm_mlp_adapter="/home/hyeonan/LLaVA/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin",
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_use_im_patch_token=False
    )
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=50000,
        save_total_limit=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        fp16=True, 
        tf32=True, 
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        output_dir="./checkpoints/llava-v1.5-7b",
        group_by_modality_length=True,
        report_to="wandb",
        lora_enable=True,
        lora_r=32,
        lora_alpha=64,
        mm_projector_lr=2e-5,
        model_max_length=512
    )
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"

    data_args, trainig_args, model, tokenizer = get_model(
        model_args=model_args,
        training_args=training_args,
        data_args=data_args,
        attn_implementation=attn_implementation,
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

    batch = {key: value.to(training_args.device) for key, value in batch.items()}
    model = model.to(training_args.device)
    
    with torch.autocast(device_type='cuda'):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        )  = model.prepare_inputs_labels_for_multimodal(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
                images=batch["images"],
                labels=batch["labels"],
                past_key_values=None,
                position_ids=None
            )
        outputs = model(            
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        loss = outputs.loss 
        print(loss)