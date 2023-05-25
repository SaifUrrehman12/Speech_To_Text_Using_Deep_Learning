# !pip install datasets==1.4.1
# !pip install transformers==4.4.0
# !pip install torchaudio
# !pip install librosa
# !pip install jiwer

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import re
train_df = pd.read_excel('/content/drive/MyDrive/Saif/ds.xlsx')

train_df["text"] = train_df.text.apply(lambda s: re.sub(r"[;'\",/?.{}+=\\|“\[\]\t:,*'َ'ُ'ِ'ْ-]|[a-z]|[A-Z]|[0-9]","",str(s)))
train_df = train_df.dropna()
train_df = train_df[~train_df.text.eq('')]
train_df = train_df.reset_index(drop=True)
train_df.columns=['path','sentence']
train_df

train_df['path'][0]

test_df = pd.read_csv("/content/drive/MyDrive/Saif/Transcript.txt",header=None)
test_df = test_df.dropna()
test_df = test_df.reset_index(drop=True)
test_df.columns=['sentence']
test_df

test_df['path'] = ""

test_df.info()

number=0
 index=0
for row in test_df['path']:
  number=number+1
  test_df['path'][index]=test_df['path'][index].replace(row,"/content/drive/MyDrive/Saif/Test Dataset/"+str(number)+".wav")
  index=index+1

test_df

test_df["sentence"] = test_df.sentence.apply(lambda s: re.sub(r"[;'\",/?.{}+=\\|“\[\]\t:,*'َ'ُ'ِ'ْ-]|[a-z]|[A-Z]|[0-9]","",str(s)))
test_df = test_df[~test_df.sentence.eq('')]
test_df = test_df.dropna()
test_df = test_df.reset_index(drop=True)

test_df

test_df['path'][0]

train_df = train_df.astype(str)
test_df=test_df.astype(str)

train_df.info()

import os
from datasets import load_dataset, load_metric
  # type(common_voice_test)
from datasets import Dataset


common_voice_train = Dataset.from_pandas(train_df)
common_voice_test=Dataset.from_pandas(test_df)

import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

#common_voice_train = common_voice_train.map(remove_special_characters)
#common_voice_test = common_voice_test.map(remove_special_characters)

#show_random_elements(common_voice_train.remove_columns(["path"]))

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

"""Now, we create the union of all distinct letters in the training dataset and test dataset and convert the resulting list into an enumerated dictionary."""

#vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {
  '|':0,
  'ا':1,
  'آ':2,
  'ب':3,
  'پ':4,
  'ت':5,
  'ٹ':6,
  'ث':7,
  'ج':8,
  'چ':9,
  'ح':10,
  'خ':11,
  'د':12,
  'ڈ':13,
  'ذ':14,
  'ر':15,
  'ڑ':16,
  'ز':17,
  'ژ':18,
  'س':19,
  'ش':20,
  'ص':21,
  'ض':22,
  'ط':23,
  'ظ':24,
  'ع':25,
  'غ':26,
  'ف':27,
  'ق':28,
  'ک':29,
  'گ':30,
  'ل':31,
  'م':32,
  'ن':33,
  'ں':34,
  'و':35,
  'ہ':36,
  'ھ':37,
  'ء':38,
  'ئ':39,
  'ی':40,
  'ے':41,
  '[UNK]':42,
  '[PAD]':43,
  '':44
}

"""Let's now save the vocabulary as a json file."""

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

"""In a final step, we use the json file to instantiate an object of the `Wav2Vec2CTCTokenizer` class."""

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

"""Next, we will create the feature extractor.

### Create XLSR-Wav2Vec2 Feature Extractor
"""

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# from google.colab import drive
# drive.mount('/content/gdrive/')

# processor.save_pretrained("/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo")

"""Next, we can prepare the dataset."""

common_voice_train[0]

import torchaudio

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

common_voice_train = common_voice_train.map(speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)

import librosa
import numpy as np

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

#common_voice_train = common_voice_train.map(resample, num_proc=4)
#common_voice_test = common_voice_test.map(resample, num_proc=4)

common_voice_train

import IPython.display as ipd
import numpy as np
import random

rand_int = random.randint(0, len(common_voice_train)-1)

ipd.Audio(data=np.asarray(common_voice_train[rand_int]["speech"]), autoplay=True, rate=16000)

rand_int = random.randint(0, len(common_voice_train)-1)

print("Target text:", common_voice_train[rand_int]["target_text"])
print("Input array shape:", np.asarray(common_voice_train[rand_int]["speech"]).shape)
print("Sampling rate:", common_voice_train[rand_int]["sampling_rate"])

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)

"""## Training"""

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
  # output_dir="./wav2vec2-large-xlsr-turkish-demo",
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=50,
  eval_steps=50,
  logging_steps=50,
  learning_rate=3e-2,
  warmup_steps=50,
  save_total_limit=2,
)

"""Now, all instances can be passed to Trainer and we are ready to start training!"""

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

"""### Training

```javascript
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```
"""

trainer.train()

model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-turkish-demo").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-large-xlsr-turkish-demo")

"""Now, we will just take the first example of the test set, run it through the model and take the `argmax(...)` of the logits to retrieve the predicted token ids."""

input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

"""We adapted `common_voice_test` quite a bit so that the dataset instance does not contain the original sentence label anymore. Thus, we re-use the original dataset to get the label of the first example."""

common_voice_test_transcription = load_dataset("common_voice", "tr", data_dir="./cv-corpus-6.1-2020-12-11", split="test")

"""Finally, we can decode the example."""

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(common_voice_test_transcription[0]["sentence"].lower())

