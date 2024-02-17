#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pickle

import torch
from torch.nn import CrossEntropyLoss
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the training data."})
    val_data_path: str = field(metadata={"help": "Path to the validation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        # sources = sources[:50]
        # targets = targets[:50]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    if data_args.val_data_path is None:
        val_dataset = None
    else:
        val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.val_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)

class TrainsetEvalCallback(TrainerCallback):
    """
    Saves perplexity to pickles after each epoch.
    """

    def __init__(self, trainer):
        super().__init__()
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        
        model = kwargs["model"]
        eval_dataloader = self._trainer.get_eval_dataloader()
        model.eval()
        
        epoch_perplexities = []

        for i, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                labels = batch["labels"]
                outputs = model(**batch)

                batch_perplexities = loss_per_sample(outputs.logits, labels)
                
                all_batch_perplexities = self._trainer.accelerator.gather_for_metrics(batch_perplexities)
                epoch_perplexities.extend(all_batch_perplexities)

        if state.is_local_process_zero:
            epoch_perplexities = epoch_perplexities[:len(self._trainer.eval_dataset)]
            print(f"Size of dataset for perplexities dump at epoch pre: {len(epoch_perplexities)}")
            with open(f'./dump-pplxs/epoch-pre.pkl', "wb") as fOut:
                pickle.dump(epoch_perplexities, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        
        model.train()


    def on_epoch_end(self, args, state, control, **kwargs):

        model = kwargs["model"]
        eval_dataloader = self._trainer.get_eval_dataloader()
        model.eval()
        
        epoch_perplexities = []

        for i, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                labels = batch["labels"]
                outputs = model(**batch)

                batch_perplexities = loss_per_sample(outputs.logits, labels)
                
                all_batch_perplexities = self._trainer.accelerator.gather_for_metrics(batch_perplexities)
                epoch_perplexities.extend(all_batch_perplexities)

        if state.is_local_process_zero:
            epoch_perplexities = epoch_perplexities[:len(self._trainer.eval_dataset)]
            print(f"Size of dataset for perplexities dump at epoch {int(state.epoch)}: {len(epoch_perplexities)}")
            with open(f'./dump-pplxs/epoch-{round(state.epoch)}.pkl', "wb") as fOut:
                pickle.dump(epoch_perplexities, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        
        model.train()

def loss_per_sample(logits, labels):

    BATCH_SIZE = logits.size(dim=0)
    VOCAB_SIZE = logits.size(dim=-1)

    # move labels to correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fn = CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1))

    # split concated batch loss values into samples
    loss_tensor_per_sample = torch.tensor_split(loss, BATCH_SIZE)

    # In the unlikely scenario that loss is exactly zero for
    # all tokens, torch.tensor(0.0) is used
    sample_loss = [torch.sum(tensor)/torch.count_nonzero(tensor)
                    if bool(torch.count_nonzero(tensor) > 0)
                    else torch.tensor(0.0).to(tensor.device)
                    for tensor in loss_tensor_per_sample]
    
    sample_perplexity = [torch.exp(s).item() for s in sample_loss]

    # labels[i] maps to sample_loss[i] and sample_perplexity[i]
    return sample_perplexity


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    print("Data Module ready!")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.add_callback(TrainsetEvalCallback(trainer))
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()