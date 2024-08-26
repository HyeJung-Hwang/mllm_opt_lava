


from typing import Dict
import torch
from dataset import CocoDataset
from llava.model.langauge_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava import conversation as conversation_lib
import transformers

from llava.train.train import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedDataset, ModelArguments, TrainingArguments
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


    # Set default_conversation
    default_conversation = conversation_lib.conv_templates["v1"]

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
    attn_implementation = "flash_attention_2"
    model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.float16 if training_args.bf16 else None),
            **{}
    )
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8,16]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
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
    batch = {key: value.to(training_args.device) for key, value in batch.items()}

    def print_module_device(module, module_name):
        try:
            device = next(module.parameters()).device
            print(f"{module_name} is on device: {device}")
        except StopIteration:
            print(f"{module_name} has no parameters.")


    # 주요 모듈들의 디바이스 확인
    print_module_device(model, "PeftModelForCausalLM")
    print_module_device(model.base_model, "base_model (LoraModel)")
    print_module_device(model.base_model.model, "model (LlavaLlamaForCausalLM)")
    print_module_device(model.base_model.model.model, "model.model (LlavaLlamaModel)")
    print_module_device(model.base_model.model.model.embed_tokens, "embed_tokens")
    print_module_device(model.base_model.model.model.layers, "layers (LlamaDecoderLayer)")
    print_module_device(model.base_model.model.model.norm, "norm (LlamaRMSNorm)")
    print_module_device(model.base_model.model.model.vision_tower, "vision_tower (CLIPVisionTower)")
    print_module_device(model.base_model.model.model.vision_tower.vision_tower, "vision_tower.vision_tower (CLIPVisionModel)")
    print_module_device(model.base_model.model.lm_head, "lm_head")
    print_module_device(model.base_model.model.model.mm_projector, "mm_projector")
    with torch.autocast(device_type='cuda'):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        )  = model.prepare_inputs_labels_for_multimodal(past_key_values=None,position_ids=None,**batch)
        outputs = model(            
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        loss = outputs.loss 