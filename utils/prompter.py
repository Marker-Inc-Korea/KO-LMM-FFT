import logging
import torch
import os

from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Sequence, Union, List

# Model values
IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_TOKEN = "<image>"

IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305] # https://github.com/AIDC-AI/Ovis/issues/21

# Data Collator
class DataCollatorForMultimodalDataset:
    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        
        pixel_values, input_ids, labels = tuple([instance[key] for instance in instances]
                                                for key in ("pixel_values", "input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id)
        
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_ID)
        
        num_valid_label = torch.not_equal(labels, IGNORE_ID).sum().item()
        
        if num_valid_label == 0:
            logging.warning(
                f'[DataCollatorForMultimodalDataset] All labels in a batch are ignored, which may lead to training instability\n{input_ids=}\n{attention_mask=}\n{labels=}')
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values
        )

# Dataset loading
class ConversationDataset(Dataset):
    def __init__(self, data, max_length, text_tokenizer, visual_tokenizer, model, eos_token_flag):
        self.samples = data
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.visual_tokenizer = visual_tokenizer
        self.text_max_length = max_length
        self.max_partitions = [9,1,1]
        
        self.eos_token_flag = eos_token_flag
        self.eos_token = 1 # 107 = <end_of_turn> / 1 = <eos>
        self.IMAGE_IGNORE_ID = [IMAGE_ATOM_ID] + IMAGE_INDICATOR_IDS
        self.IGNORE_ID = IGNORE_ID
         
    def read_image(self, path):
        try:
            full_path = os.path.join(self.image_dir, path)
            image = Image.open(full_path).convert('RGB')
            return image, None
        except Exception as e:
            return None, e
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        
        #print(self.samples[i])
        conversations = self.samples[i]['instruction']
        query = f'<image>\n{conversations}' # if image, <image> add
        output = self.samples[i]['output']

        images = None
        max_partition = None
        if 'image' in self.samples[i]:
            images = []
            image_paths = self.samples[i]['image']
            for image_path in image_paths:
                image, e = self.read_image(image_path)
                if image is None:
                    logging.warning(
                        f'reading image failed with index: {i}, image path: {image_path}, and exception: {e}')
                    images = None
                    break
                images.append(image)
            max_partition = self.max_partitions[0] if len(images) == 1 else self.max_partitions[1]
            
        if images:
            max_partition = self.max_partitions[0] if len(images) == 1 else self.max_partitions[1]
            
        if self.eos_token_flag:
            output = output + '<eos>' # <eos> or <end_of_turn>

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, # conversations, query
            images,
            max_partition=max_partition,
            generation_preface=output, # generation output
            return_labels=False,
            propagate_exception=False # image none okay
        )
        
        #print(pixel_values) ## if <image>, all zero tensor
        if pixel_values is None:
            pixel_values, _ = self.visual_tokenizer.mock_input()
        
        # Avoid error (label cannot calculate loss)
        for token_id in self.IMAGE_IGNORE_ID:
            input_ids = torch.where(input_ids==token_id, self.IGNORE_ID, input_ids)

        input_ids = input_ids[:self.text_max_length]
        labels = input_ids.clone() # LLM transformer
        
        ## examples
        #print(prompt)
        
        #print(input_ids)
        
        #decode = self.text_tokenizer.decode(input_ids)
        #print(decode)
        
        #decode = self.text_tokenizer.decode(labels)
        #print(decode)
        
        #print(self.text_tokenizer.decode([2, 106, 1645, 108])) # <bos><start_of_turn>user\n
        #print(self.text_tokenizer.decode([108])) # \n

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )