import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
from librosa.util import pad_center
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, TrainerCallback, PrinterCallback, AutoFeatureExtractor, SequenceFeatureExtractor
import evaluate
from torch.utils.data import random_split
import copy
from accelerate import Accelerator
# MASTER CODE



class AudioArrays(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio files. (BabbleCor_clips)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.audios_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base") # load feature extractor

    def __len__(self):
        return len(self.audios_df)

    # __getitem__ to support the indexing such that dataset[i] can be used to get the ith sample.
    def __getitem__(self, row_index):

        get_clip = os.path.join(self.root_dir, self.audios_df.iloc[row_index, 0]) # get row_index row in column 0 (clip_id)

        audio_array, sr = torchaudio.load(get_clip) # get original sample rate of audio file

        # Create a Resample transform
        resample_transform = Resample(orig_freq=sr, new_freq=16000)

        # Apply the transform to the audio waveform
        resampled_waveform = resample_transform(audio_array)

        # pad 
        if (resampled_waveform.size())[1] != 8000:
            # Pad it to 8000 if needed
            leftover = 8000 - (resampled_waveform.size())[1]
            half = leftover // 2 
            #left_half = half
            
            if leftover % 2 == 0:
                resampled_waveform = torch.nn.functional.pad(resampled_waveform, (half, half), value=0)
            else:
                resampled_waveform = torch.nn.functional.pad(resampled_waveform, (half + 1, half), value=0)

        # stereo to mono if needed
        if (resampled_waveform.size())[0] != 1:
            mono_waveform = torch.mean(resampled_waveform, dim=0)
            resampled_waveform = mono_waveform
        else: # what does this even do?
            resampled_waveform = resampled_waveform.reshape(8000)

        # featurize
        '''
        from hugging face:
        def preprocess_function(examples):
            # audio_arrays = [x["array"] for x in examples["audio"]]
            inputs = feature_extractor(
                audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True # change sampling_rate to 16000, change max_length to 8000, add padding argument
            )
            return inputs
        '''
        inputs = self.feature_extractor(resampled_waveform, sampling_rate=16000, max_length=16000, truncation=True, padding=True)['input_values'][0]
        #print(inputs)

        # get label
        maturity_label = self.audios_df.iloc[row_index, 1]
        # maturity_class = 0

        '''if maturity_label == 'Canonical':
            maturity_class = 0
        elif maturity_label == 'Non-canonical':
            maturity_class = 1
        elif maturity_label == 'Other':
            maturity_class = 2'''

        sample = {'input_values': inputs, 'label': maturity_label} # feature and label

        if self.transform:
            sample = self.transform(sample)

        return sample


def test_custom(transformed_dataset):
    '''
    transformed_dataset = AudioArrays(csv_file='../Audio_Clips_Balanced.csv',
                                           root_dir='../BabbleCor_clips')
    '''

    for i, sample in enumerate(transformed_dataset):
        print(i, len(sample['input_values']), type(sample['input_values']), sample['label'])
        # shape of tensor is [channels, samples]
        print((sample['input_values'])[0]) # yup, prints out some non zero number!

        #if (sample['input_values'].size()[0] != 1):
        #    print(i, sample['input_values'].size(), type(sample['input_values']), sample['label'])

        if i == 3:
            break



def return_custom_set():
    transformed_dataset = AudioArrays(csv_file='../Audio_Clips_ID_2Classes.csv',
                                           root_dir='../BabbleCor_clips')

    return transformed_dataset




'''
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
'''

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


def train_model(train_dataset, test_dataset):
    """ id2label = {'0': 'Canonical', '1': 'Non-canonical', '2': 'Other'}
    label2id = {'Canonical': 0, 'Non-canonical': 1, 'Other': 2} """
    id2label = {'0': 'Canonical', '1': 'Non-canonical'}
    label2id = {'Canonical': 0, 'Non-canonical': 1}

    model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=2, label2id=label2id, id2label=id2label
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    training_args = TrainingArguments(
    output_dir="my_model_1_24",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5, # reduce learning rate from 3e-5? 
    per_device_train_batch_size=32, # increase batch size from 32?
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32, # increase batch size from 32?
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    #dataloader_num_workers = ___

    #push_to_hub=True,
    )

    # accelerate launch --config_file {/Users/your_username/Library/Caches/default_config.yaml} build_parallel_trainer.py
    accelerator = Accelerator()
    trainer = accelerator.prepare(Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor, # not necessary, probably not causing the issue
        compute_metrics=compute_metrics,
    ))
    trainer.add_callback(PrinterCallback) 

    trainer.train()


def main():
    # test_custom()

    custom_dataset = return_custom_set()
    #test_custom(custom_dataset)
    #return

    # split into train and test
    train_size = int(0.8 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

    train_model(train_dataset, test_dataset)
    



if __name__ == "__main__":
    main()