import torch
from torch.utils.data import DataLoader
from data_util import MimicFullDataset, my_collate_fn, aweful_collate_function

from torchinfo import summary
import logging
import argparse
import json
from train_utils import generate_model
from evaluation import all_metrics, print_metrics
from find_threshold import find_threshold_micro
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from find_threshold import find_threshold_micro

from explain import ICDModelInterpreter
from cosine import calculate_cosine_similarity

from tqdm import tqdm
from eval_utils import get_words_from_ids
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup

import numpy as np

def cosine_sim(model, device,dataset, dataloader):

    model.calculate_label_hidden()

    batch = next(iter(dataloader))

    batch_gpu = tuple([x.to(device) for x in batch])

    input_word, word_mask = batch_gpu[0:2]

    label_embed = model.label_feats

    hidden = model.calculate_text_hidden(input_word, word_mask)

    it = tqdm(dataloader)
    interpreter = ICDModelInterpreter(model, device, dataset)

    code_desc = {}
    ind_desc = dataset.extract_label_desc(dataset.ind2c)

    for ind, code in dataset.ind2c.items():
        code = dataset.ind2c[ind]
        code_desc[code] = ind_desc[ind]


    mean_prediction_sim = {}

    k = 5

    
    method_sim_scores = {}
    method_sim_scores['attn'] = []
    method_sim_scores['ig'] = []
    for batch in it:
        attn_dict = interpreter.interpret(batch)
        batch_gpu = tuple([x.to(device) for x in batch])
        input_word, word_mask = batch_gpu[0:2]

        hidden = model.calculate_text_hidden(input_word, word_mask)

        mean_code_sim = {}
        for code, data in attn_dict.items():
            code_ind = dataset.c2ind[code]

            #get indicies of top-k entries with highest attention scores using torch
            top_k_indices = torch.topk(torch.from_numpy(data[0].word_attributions), k=k).indices
            
            text = data[0].raw_input_ids #get the text

            #get the top-k words
            try:
                top_k_words = [text[i] for i in top_k_indices]
            except:
                logger.info(f"Top k indices: {top_k_indices}")
                logger.info(f"Text: {text}")
                logger.info(f"Length of text: {len(text)}")
                #logger.info(f"Length of word attributions: {len(data[0].word_attributions)}")
                break
            
            top_k_encoded = [input_word[0][i] for i in top_k_indices]
            top_k_masks = [word_mask[0][i] for i in top_k_indices]
            
            top_k_embeddings  = [hidden[0][i] for i in top_k_indices]

            target_sim_scores = []
            for target in top_k_embeddings:
                target = target.unsqueeze(0)
                sim = calculate_cosine_similarity(target, label_embed, softmax=False)
                
                target_sim_scores.append(sim[code_ind].item())


            mean_code_sim[code] = sum(target_sim_scores) / len(target_sim_scores)
            #logger.info(f"Mean code similarity:{code} - {code_desc[code]}: {mean_code_sim[code]}")
        #logger.info(f"Mean code similarity: {mean_code_sim.values()}")
        method_sim_scores['attn'].extend(mean_code_sim.values())
    average_score = sum(method_sim_scores['attn']) / len(method_sim_scores['attn'])
    logger.info(f"Average attention similarity score: {average_score}")

def word_matching(model, device, dataset, dataloader, scoring_method='attn'):
    model.calculate_label_hidden()

    batch = next(iter(dataloader))

    batch_gpu = tuple([x.to(device) for x in batch])

    input_word, word_mask = batch_gpu[0:2]

    it = tqdm(dataloader)
    interpreter = ICDModelInterpreter(model, device, dataset)

    code_desc = {}
    ind_desc = dataset.extract_label_desc(dataset.ind2c)

    for ind, code in dataset.ind2c.items():
        code = dataset.ind2c[ind]
        code_desc[code] = ind_desc[ind]

    k = 5

    code_to_words = {}

    for batch in it:

        if(scoring_method == 'attn'):
            score_dict = interpreter.get_attention_scores(batch)
        elif(scoring_method == 'ig'):
            score_dict = interpreter.interpret(batch)
        else:
            logger.info("Invalid scoring method")
            return

        batch_gpu = tuple([x.to(device) for x in batch])
        
        for code, data in score_dict.items():

            #get indicies of top-k entries with highest attention scores using torch
            top_k_indices = torch.topk(torch.from_numpy(data[0].word_attributions), k=k).indices
            
            text = data[0].raw_input_ids #get the text

            #get the top-k words
            try:
                top_k_words = [text[i] for i in top_k_indices]
            except:
                logger.info(f"Top k indices: {top_k_indices}")
            
            code_to_words.setdefault(code, []).extend(top_k_words)
        
    with open(f'code_top_words_rep_{scoring_method}.json', 'w') as f:
        json.dump(code_to_words, f)
    
def count_matches(method='attn', repetition = True, abbreviation = False):
    #load abbreviations
    with open('clean_abbreviations.json', 'r') as f:
        abbreviations = json.load(f)

    #load synonyms
    with open('synonyms.json', 'r') as f:
        syn = json.load(f)

    #load top words from attn
    if repetition:
        rep = "_rep"
    else:
        rep = ""

    if abbreviation:
        abbr = "_abbr"
    else:
        abbr = ""

    
    with open(f'code_top_words{rep}_{method}.json', 'r') as f:
        top_words_attn = json.load(f)

    code_to_accuracy = {}
    all_words = {}
    for code in top_words_attn.keys():
        words = top_words_attn[code]
    
        matches = 0
        for word in words:
            if word in syn[code].split(" "):
                matches += 1

        print(f"Code: {code} - Matches: {matches} - Total: {len(words)}")
        accuracy = matches / len(words)
        code_to_accuracy[code] = accuracy
        all_words["matches"] = all_words.get("matches", 0) + matches
        all_words["total"] = all_words.get("total", 0) + len(words)
    
    for code, accuracy in code_to_accuracy.items():
        print(f"Code: {code} - Accuracy: {accuracy}")
    #calculate average accuracy
    all_words["average"] = all_words["matches"] / all_words["total"]
    code_to_accuracy['mean'] = all_words["average"]
    #average_accuracy = sum(code_to_accuracy.values()) / len(code_to_accuracy)
    print(f"Average accuracy: {code_to_accuracy['mean']}")

    #save to a json file
    with open(f'code_accuracy{rep}{abbr}_{method}_weighted.json', 'w') as f:
        json.dump(code_to_accuracy, f)

def save_synonyms(model, device, dataset, dataloader):
    model.calculate_label_hidden()


    it = tqdm(dataloader)
    interpreter = ICDModelInterpreter(model, device, dataset)

    code_desc = {}
    ind_desc = dataset.extract_label_desc(dataset.ind2c)

    for ind, code in dataset.ind2c.items():
        code = dataset.ind2c[ind]
        code_desc[code] = ind_desc[ind]

    for code in code_desc.keys():
        logger.info(f"Code: {code} - {code_desc[code]}")

    logger.info(f"{model.c_input_word} - {model.c_input_word.shape}")
    
    decoded_synonyms = {}

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    logger.info(f"Stopwords: {stop_words}")
    
    for i, code in enumerate(code_desc.keys()):
        synonyms = []
        print(f"{i*8} - {(i*8)+8}")
        for j in range(i*8, (i*8)+8):
            
            
            curr_syn_encoded = model.c_input_word[j].detach().cpu().numpy()
            curr_syn_decoded = get_words_from_ids(curr_syn_encoded, dataset.word2id)

            #remove padding words
            curr_syn_decoded = [word for word in curr_syn_decoded if (word != "**PAD**" and word not in stop_words and word != "**UNK**")]
            #remove duplicates
            curr_syn_decoded = list(set(curr_syn_decoded))
            
            
           
            synonyms.extend(curr_syn_decoded)
        
        #logger.info(f"Synonyms: {synonyms}")
        current_synonym = " ".join(list(set(synonyms)))
        decoded_synonyms[code] = current_synonym
        
    
        logger.info(f"\nCode: {code} - {code_desc[code]}\n")
        logger.info(f"Synonyms: {decoded_synonyms[code]}")
    
    k = 5

    with open('synonyms.json', 'w') as f:
        json.dump(decoded_synonyms, f)
    
def get_abbreviations():
    response = requests.get("https://www.asha.org/practice-portal/professional-issues/documentation-in-health-care/common-medical-abbreviations/")
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table')

    if table is None:
        logger.info("Table not found")
        return None
    
    rows = []
    for row in table.find_all('tr'):
        columns = []
        for col in row.find_all('td'):
            columns.append(col.get_text(strip=True))
        rows.append(columns)

    #remove alphabetical headers
    rows = [row for row in rows if len(row) > 1]

    #lowercase all words
    rows = [[word.lower() for word in row] for row in rows]
    
    return rows

def clean_abbreviations(file_path):
    
    #load from json
    with open(file_path, 'r') as f:
        abbr = json.load(f)

    clean = []
    for row in abbr:
        short = row[0]
        full = row[1]

        short_words = short.split(",")
        full_words = full.split(",")

        if(len(short_words) == len(full_words) == 1):
            clean.append(row)
            continue
        
        if len(short_words) == len(full_words):
            for i in range(len(short_words)):
                short_word = short_words[i]
                full_word = full_words[i]

                clean.append([short_word, full_word])
            continue
        
        if len(short_words) > len(full_words):
            for i in range(len(short_words)):
                short_word = short_words[i]
                full_word = full

                clean.append([short_word, full_word])
            continue

        if len(short_words) < len(full_words):
            for i in range(len(full_words)):
                for j in range(len(short_words)):
                    short_word = short_words[j]
                    full_word = full_words[i]

                    clean.append([short_word, full_word])
            continue
    
    #remove punctuation
    clean = [[word.replace(".", "").replace(",", "").replace(";", "").replace(":", "").replace("/", "").replace(")", "").replace("(", "") for word in row] for row in clean]
    
    #remove spaces at the beginning and end of words
    clean = [[word.strip() for word in row] for row in clean]

    #remove duplicate rows
    clean = list(set([tuple(row) for row in clean]))
    return clean

def fetch_and_save_abbreviations():
    table = get_abbreviations()
    
    abbr = clean_abbreviations('raw_abbreviations.json')
    #save as json
    with open('clean_abbreviations.json', 'w') as f:
        json.dump(abbr, f)


def attribution_score_correlation(model, device, dataset, dataloader, scoring_method='attn'):
    model.calculate_label_hidden()

    

    it = tqdm(dataloader)
    interpreter = ICDModelInterpreter(model, device, dataset)

    code_desc = {}
    ind_desc = dataset.extract_label_desc(dataset.ind2c)

    for ind, code in dataset.ind2c.items():
        code = dataset.ind2c[ind]
        code_desc[code] = ind_desc[ind]

    k = 5

    code_to_words = {}

    with open('synonyms.json', 'r') as f:
        syn = json.load(f)

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    for batch in it:

        if(scoring_method == 'attn'):
            score_dict = interpreter.get_attention_scores(batch)
        elif(scoring_method == 'ig'):
            score_dict = interpreter.interpret(batch)
        else:
            logger.info("Invalid scoring method")
            return

       
        
        for code, data in score_dict.items():

           
            word_attributions = data[0].word_attributions
            text = data[0].raw_input_ids #get the text

            #remove stopwords, padding and unknown words

            curr = {}

            for i, word in enumerate(text):
                if (word in stop_words or word == "**PAD**" or word == "**UNK**"):
                    continue
                
                curr.setdefault("word", []).append(word)
                curr.setdefault("score", []).append(str(word_attributions[i]))

                if word in syn[code].split(" "):
                    curr.setdefault("match", []).append(1)
                else:
                    curr.setdefault("match", []).append(0)
                
            code_to_words.setdefault(code, []).append(curr)
            
            
    with open(f'code_match_pred_{scoring_method}_notewise.json', 'w') as f:
        json.dump(code_to_words, f)  
           
            
def calculate_auc_metric(scoring_method = "attn"):
    
    #load code match predictions
    with open(f'code_match_pred_{scoring_method}.json', 'r') as f:
        code_to_words = json.load(f)
    
    all_words = {}
    metrics = {}
    for code, data in code_to_words.items():
        words = data['word']
        scores = data['score']
        matches = data['match']

        #convert scores to floats
        scores = [float(score) for score in scores]

        #calculate auc
        auroc = roc_auc_score(matches, scores)
        auprc = average_precision_score(matches, scores)
        threshold = find_threshold_micro(np.array(scores), np.array(matches))
        f1 = f1_score(np.array(matches), [1 if score > threshold else 0 for score in scores])

        #add words from codes to dictionary
        all_words.setdefault("words", []).extend(words)
        all_words.setdefault("scores", []).extend(scores)
        all_words.setdefault("matches", []).extend(matches)

        metrics[code] = {"auroc": auroc, "auprc": auprc, "f1": f1, "threshold": threshold}
        print(f"Code: {code} - {metrics[code]}")
    
    #calculate overall metrics
    all_auroc = roc_auc_score(all_words['matches'], all_words['scores'])
    all_auprc = average_precision_score(all_words['matches'], all_words['scores'])
    all_threshold = find_threshold_micro(np.array(all_words['scores']), np.array(all_words['matches']))
    all_f1 = f1_score(all_words['matches'], [1 if score > all_threshold else 0 for score in all_words['scores']])

    metrics['all'] = {"auroc": all_auroc, "auprc": all_auprc, "f1": all_f1, "threshold": all_threshold}
    print(f"All Words: {metrics['all']}")

    #save metrics to a json file
    with open(f'metrics_{scoring_method}.json', 'w') as f:
        json.dump(metrics, f)
              

def compare_summarisation(model, device, dataset, dataloader, scoring_method='attn', summary_type='summary'):
    model.calculate_label_hidden()

    it = tqdm(dataloader)
    interpreter = ICDModelInterpreter(model, device, dataset)

    code_desc = {}
    ind_desc = dataset.extract_label_desc(dataset.ind2c)

    for ind, code in dataset.ind2c.items():
        code = dataset.ind2c[ind]
        code_desc[code] = ind_desc[ind]

    k = 5

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    code_to_accuracy = {}
    all_codes = {}
    for batch_text, batch_sum in it:

        if(scoring_method == 'attn'):
            score_dict = interpreter.get_attention_scores(batch_text)
        elif(scoring_method == 'ig'):
            score_dict = interpreter.interpret(batch_text)
        else:
            logger.info("Invalid scoring method")
            return

        for code, data in score_dict.items():

            #get indicies of top-k entries with highest attention scores using torch
            top_k_indices = torch.topk(torch.from_numpy(data[0].word_attributions), k=30).indices
            
            text = data[0].raw_input_ids #get the text

            #get the top-k words
            try:
                top_k_words = [text[i] for i in top_k_indices]
            except:
                logger.info(f"Top k indices: {top_k_indices}")

            unique_words = []

            for word in top_k_words:
                if word not in unique_words and len(unique_words) < k:
                    unique_words.append(word)

            sum_word, sum_mask = batch_sum[0:2]
        
            sum_text = get_words_from_ids(np.array(sum_word[0]), dataset.word2id)

            
            #remove stopwords, padding and unknown words
            sum_text = [word for word in sum_text if (word not in stop_words and word != "**PAD**" and word != "**UNK**")]

            matches = 0
            for target_word in unique_words:
                if target_word in sum_text:
                    matches += 1

            
            code_to_accuracy.setdefault(code, {})["matches"] = code_to_accuracy.setdefault(code, {}).get("matches", 0) + matches
            code_to_accuracy.setdefault(code, {})["total"] = code_to_accuracy.setdefault(code, {}).get("total", 0) + k
    
    total_matches = 0
    grand_total = 0
    for code, data in code_to_accuracy.items():
        matches = data["matches"]
        total = data["total"]

        total_matches += matches
        grand_total += total

        accuracy = matches / total
        code_to_accuracy[code]["accuracy"] = accuracy
       
    code_to_accuracy["mean"] = {}
    code_to_accuracy["mean"]["matches"] = total_matches
    code_to_accuracy["mean"]["total"] = grand_total
    code_to_accuracy["mean"]["accuracy"] = total_matches / grand_total
    
    #calculate average accuracy
    code_to_accuracy["mean"]["accuracy"] = code_to_accuracy["mean"]["matches"] / code_to_accuracy["mean"]["total"]
    print(f"Average accuracy: {code_to_accuracy['mean']['accuracy']}")
    
    with open(f'summary_acc_{summary_type}_{scoring_method}_weighted_unique.json', 'w') as f:
        json.dump(code_to_accuracy, f)



if __name__ == "__main__":
    #Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=bool, default=True)
    
    args = parser.parse_args()

    #Setting up logger
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    current_log_level = log_levels[min(len(log_levels) - 1, args.verbose)]
    logging.basicConfig(level=current_log_level)
    logger = logging.getLogger("eval_interpreter")

    if args.model_path.find('mimic3-50') >= 0:
        version = 'mimic3-50'
    else:
        version = 'mimic3'

    logger.info(f"Version: {version}")
    logger.info(f"Model Path: {args.model_path}")
    
    if args.gpu:
        device = "cuda:0"
    else:
        device = "cpu"
    
    #Loading model
    model = torch.load(args.model_path).to(device)
    model.eval()
    model = model.to(device)

    logger.debug(summary(model))

    #Setting up dataset and dataloader
    word_embedding_path = '/cs/student/msc/dsml/2023/mdavudov/UCB/ICD-MSMN/embedding/word2vec_sg0_100.model'

    PAD_CHAR = "**PAD**"
    PAD_INDEX = 150695

    for summary_type in ["summary_chunk_50_addition", "summary_extractive_addition", "summary", "summary_chunk_50", "summary_extractive"]:
        for scoring_method in ['ig', 'attn']:
            
             dataset = MimicFullDataset(version, "test", word_embedding_path, 4000, summarised=True, summary_name=summary_type)
             dataloader = DataLoader(dataset, batch_size=1, collate_fn=aweful_collate_function, shuffle=False, num_workers=1)
             compare_summarisation(model, device, dataset, dataloader, scoring_method=scoring_method, summary_type=summary_type)

    
