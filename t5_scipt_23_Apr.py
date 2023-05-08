# import necessary libraries
import json
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, get_scheduler
# from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import re
import sys
# sys.path.insert(1, '/kaggle/working/metrics_recipe_updated.py')
import metrics_recipe_updated
import matplotlib.pyplot as plt


def validate(val_dataloader, model, tokenizer, device):
    print("Validating...")
    for batch in tqdm(val_dataloader):
        generated_ids = model.module.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            max_length=513,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )  
        preds = [
                tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for gen_id in generated_ids
        ]
        for i, pred in enumerate(preds):
            output_text = "".join(pred)
            recipe_id = batch['id'][i]
            x = re.split('Title: | Recipe: ', output_text)
        
            if len(x) < 3:
                with open(f'./val_data/GT/instructions/{recipe_id}.txt', 'w') as file:
                    file.write('-1')
                with open(f'./val_data/Predicted/instructions/{recipe_id}.txt', 'w') as file:
                    file.write('-1')
            else:
                predicted_title, predicted_recipe = x[1], x[2]
                predicted_recipe = predicted_recipe.replace('. ','.\n')
                with open(f'./val_data/Predicted/instructions/{recipe_id}.txt', 'w') as file:
                    file.write(predicted_recipe)
                
                actual_recipe = batch['recipe'][i]
                with open(f'./val_data/GT/instructions/{recipe_id}.txt', 'w') as file:
                    file.write(actual_recipe)
                
            # title = batch['title'][i]
            # ingredients = batch['ingredients'][i].replace(', ','\n')
            # with open(f'./val_data/GT/title/{recipe_id}.txt', 'w') as file:
            #     file.write(title)
            # with open(f'./val_data/Predicted/title/{recipe_id}.txt', 'w') as file:
            #     file.write(title)
            # with open(f'./val_data/GT/ingredients/{recipe_id}.txt', 'w') as file:
            #     file.write(ingredients.lower())
            # with open(f'./val_data/Predicted/ingredients/{recipe_id}.txt', 'w') as file:
            #     file.write(ingredients.lower())
    
    ret_metrics = metrics_recipe_updated.evaluate_metrics('val_data')
    print("Done validating")
    return ret_metrics

class RecipeDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        mode,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        self.mode = mode
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['input']
        
        

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        summary_encoding = self.tokenizer(
            data_row['output'],
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return_dict = dict(
            input_ids=text_encoding['input_ids'].flatten(),
            attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            decoder_attention_mask=summary_encoding['attention_mask'].flatten()
        )
        
        if self.mode == 'val' or self.mode == 'test':
            return_dict['id'] = data_row['id']
            return_dict['title'] = data_row['title']
            return_dict['ingredients'] = data_row['ingredients']
            return_dict['recipe'] = data_row['recipe']

        return return_dict




def load_data(tokenizer):
    #extract data from JSON files
    with open('./dataset/det_ingrs_small_updated.json') as json_file:
        ingredients_dictionary = json.load(json_file)
        
    with open('./dataset/layer1_small_updated.json') as json_file:
        layer_dictionary = json.load(json_file)

    ingredients_by_id = {}
    for item in ingredients_dictionary:
        id = item['id']
        ingredients = ', '.join([x['text'] for x in item['ingredients']])
        ingredients_by_id[id] = ingredients
    
    train_dicts = []
    val_dicts = []
    test_dicts = []
    for item in layer_dictionary:
        temp_dict = {}
        temp_dict['id'] = item['id']
        temp_dict['title'] = item['title']
        temp_dict['ingredients'] = ingredients_by_id[item['id']]
        temp_dict['recipe'] = ' \n'.join([x['text'] for x in item['instructions']])
        temp_dict['output'] = f"Title: {item['title']} \nRecipe: {temp_dict['recipe']}"
        temp_dict['input'] = f"Title: {item['title']} \nIngredients: {temp_dict['ingredients']}"
    #     \nIngredients with qty: {', '.join([x['text'] for x in item['ingredients']])}
        if item['partition'] == 'train':
            temp_dict['input'] += f"\nIngredients with quantity: {', '.join([x['text'] for x in item['ingredients']])}"
            train_dicts.append(temp_dict)
        elif item['partition'] == 'test':
            test_dicts.append(temp_dict)
        elif item['partition'] == 'val':
            val_dicts.append(temp_dict)

    train_df = pd.DataFrame(train_dicts)
    test_df = pd.DataFrame(test_dicts)
    val_df = pd.DataFrame(val_dicts)
    print(len(train_df), len(test_df), len(val_df))

    train_dataset = RecipeDataset(data = train_df, mode = 'train', tokenizer = tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10)

    test_dataset = RecipeDataset(data = test_df, mode = 'test', tokenizer = tokenizer)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=10)

    val_dataset = RecipeDataset(data = val_df, mode = 'val', tokenizer = tokenizer)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=10)

    return train_dataloader, test_dataloader, val_dataloader

def load_model_tokenizer():
    # initialize the tokenizer and model
    torch.cuda.empty_cache()
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=50)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    config = T5Config(
    vocab_size = tokenizer.vocab_size,
    pad_token_id = tokenizer.pad_token_id,
    eos_token_id = tokenizer.eos_token_id,
    decoder_start_token_id = tokenizer.pad_token_id,
    d_model = 300
    )

    model = T5ForConditionalGeneration(config)

    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return model, tokenizer, device

def train(model, tokenizer, device, train_dataloader, val_dataloader):
    num_epochs = 50
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))


    minLoss = 10000000
    optimizer = AdamW(model.parameters(), lr = 3e-4)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    print(device)
    train_loss_history = []
    val_loss_history = []
    val_metrics_history = []
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for batch in train_dataloader:
    #     for i in range(30):
    #         batch = next(iter(train_dataloader))
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
    #         print(torch.sum(outputs.loss))
            
            loss = torch.sum(outputs.loss)
    #         print(loss)
            train_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()
            progress_bar.update()
        
        train_loss /= len(train_dataloader)
        train_loss_history.append(train_loss)
        #validate
        ret_metrics = validate(val_dataloader, model, tokenizer, device)
        val_metrics_history.append(ret_metrics)
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_dataloader):
                forward_keys = {key: batch[key].to(device) for key in ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]}
                outputs = model(**forward_keys)
                logits = outputs.logits
                loss = torch.sum(outputs.loss)
    #             print(f'val loss: {loss}')
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
            print(f"Validation loss = {val_loss}")
            val_loss_history.append(val_loss)
            
        if loss < minLoss:
            print("model saved with loss = ", loss)
            torch.save(model.state_dict(), "./dataset/t5_recipe_with_trick.pt")
            minLoss = loss
        
            
        print(f'epoch: {epoch + 1} -- loss: {loss}')

    
    plt.plot(train_loss_history, label='train_loss')
    plt.plot(val_loss_history,label='val_loss')
    plt.legend()
    plt.show
    plt.savefig('./dataset/loss_graph.png')
    return model, train_loss_history, val_loss_history, val_metrics_history

def load_best_model(model, device, path = './dataset/t5_recipe_april_20.pt'):
    #LOADING MODEL
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(path))
    return model

def main():

    os.system('rm -r  test_data')
    os.system('rm -r  val_data')
    os.system('mkdir -p test_data/GT/ingredients test_data/GT/title test_data/GT/instructions')
    os.system('mkdir -p val_data/GT/ingredients val_data/GT/title val_data/GT/instructions')
    os.system('mkdir -p test_data/Predicted/ingredients test_data/Predicted/title test_data/Predicted/instructions')
    os.system('mkdir -p val_data/Predicted/ingredients val_data/Predicted/title val_data/Predicted/instructions')
    
    
    model, tokenizer, device = load_model_tokenizer()
    # model = load_best_model(model, device)

    train_dataloader, test_dataloader, val_dataloader = load_data(tokenizer)

    # ret_metrics = validate(val_dataloader, model, tokenizer, device)
    # print(ret_metrics)

    model, train_loss_history, val_loss_history, val_metrics_history = train(model, tokenizer, device, train_dataloader, val_dataloader)

    print(f"Train Loss History: {train_loss_history}")
    print(f"Val Loss History: {val_loss_history}")
    print(f"Val Metrics History: {val_metrics_history}")
    print("Training complete!")





if __name__ == "__main__":
    main()