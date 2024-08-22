from PIL import Image
from typing import Any


class VisionPreprocessor():
    def __init__(
        self,
        image_aspect_ratio: str,
        processor: Any,
        ) -> None:
        self.image_aspect_ratio = image_aspect_ratio
        self.processor = processor
        pass
    
    def __call__(
        self,
        image
        ):
        if self.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in self.processor.image_mean))
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        return image


class TextPreprocessor():
    def __init__(self) -> None:
        self.DEFAULT_IMAGE_TOKEN = "<image>"

    def __call__(self, is_multimodal,sources) -> Any:
 
        if not is_multimodal:
            return sources

        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)