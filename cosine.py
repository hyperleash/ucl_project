import torch
from torch.utils.data import DataLoader
from data_util import MimicFullDataset, my_collate_fn

from torchinfo import summary
import logging
import argparse
import json
from train_utils import generate_model
from evaluation import all_metrics, print_metrics
from find_threshold import find_threshold_micro

import numpy as np

def pool_scores(scores, window_size=8, softmax = True):
    """
    Performs max-pooling with a step size and normalizes the scores.

    Args:
        scores (np.ndarray): Array of similarity scores.
        window_size (int, optional): Window size for max-pooling. Defaults to 8.

    Returns:
        np.ndarray: Array of normalized scores after max-pooling.
    """
    # Get the number of windows and padding size
    num_windows = (len(scores) + window_size - 1) // window_size
    padding_size = window_size * num_windows - len(scores)

    # Pad the scores with zeros
    padded_scores = np.pad(scores, (0, padding_size), mode='constant')

    # Reshape into windows and perform max pooling
    pooled_scores = np.max(padded_scores.reshape(-1, window_size), axis=1)

    if softmax:
        pooled_scores = torch.softmax(torch.from_numpy(pooled_scores), dim=0)

    return pooled_scores

import torch.nn.functional as F

def calculate_cosine_similarity(text_embeddings, label_embeddings, softmax = True):
    """
    Calculates cosine similarity between text embeddings and label embeddings.

    Args:
        text_embeddings (torch.Tensor): Tensor of shape (seq_length, embedding_dim)
        label_embeddings (torch.Tensor): Tensor of shape (num_labels, embedding_dim)

    Returns:
        torch.Tensor: Tensor of shape (num_labels,) containing cosine similarities
    """

    
    # Normalize the embeddings to unit length
    text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)  # Normalize along the embedding dimension
    label_embeddings_norm = F.normalize(label_embeddings, p=2, dim=-1)

    # Calculate cosine similarity
    cosine_similarities = torch.matmul(label_embeddings_norm, text_embeddings_norm.transpose(0, 1))
    
    # Aggregate similarities along the sequence dimension 
    aggregated_similarities = torch.mean(cosine_similarities, dim=1)  # Mean pooling over sequence

    normalized_scores = pool_scores(aggregated_similarities.cpu().detach().numpy(), window_size=8, softmax=softmax)
    return normalized_scores


def get_words_from_ids(id_list, word_to_id_dict):
        id_to_word_dict = {v: k for k, v in word_to_id_dict.items()}  # Reverse the dictionary
        return [id_to_word_dict[id] for id in id_list]


def predict_cosine(dataloader, device, threshold = None, tqdm_bar = False, args = None):
    yhat, y, yhat_raw = [], [], []
    for batch in dataloader:
        batch_gpu = tuple([x.to(device) for x in batch])
        input_word, word_mask = batch_gpu[0:2]
        label_embed = model.label_feats
        hidden = model.calculate_text_hidden(input_word, word_mask)
        cosine_similarities = calculate_cosine_similarity(hidden[0], label_embed)
        
        if(threshold is None):
            threshold = args.prob_threshold

        yhat_raw.append(cosine_similarities)
        yhat.append(np.where(cosine_similarities > threshold, 1, 0))
        y.append(batch[3].cpu().detach().numpy())
    return np.vstack(yhat), np.vstack(y), np.vstack(yhat_raw)

def eval_func_cosine(dataloader, device, threshold = None, tqdm_bar = False, args = None):
    yhat, y, yhat_raw = predict_cosine(dataloader, device, threshold, tqdm_bar, args)
    if threshold is None:
        threshold = find_threshold_micro(yhat_raw, y)
    yhat = np.where(yhat_raw > threshold, 1, 0)
    metric = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
    return metric, (yhat, y, yhat_raw), threshold

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
    logger = logging.getLogger("cosine")

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

    with open("/cs/student/msc/dsml/2023/mdavudov/UCB/ICD-MSMN/output/mimic3-50_word-0.2_lstm-1-512-0.1_MultiLabelMultiHeadLAATV2-512-xav-0.2-8_max-est1-8-random_ce-0.0-1.0_bsz16-AdamW-1-4000-warm0.0-wd0.01-0.0005-rdrop5.0/args.json", 'r') as json_file:
        args_dict = json.load(json_file)

    model_args = argparse.Namespace(**args_dict)


    word_embedding_path = '/cs/student/msc/dsml/2023/mdavudov/UCB/ICD-MSMN/embedding/word2vec_sg0_100.model'

    model = generate_model(model_args, MimicFullDataset(version, "dev", word_embedding_path, 4000, summarised=False))#torch.load(model_path).to(device)#torch.load(model_path).to(device)
    
    #Loading model
    model = torch.load(args.model_path).to(device)
    model.eval()
    model = model.to(device)

    logger.debug(summary(model))

    #Setting up dataset and dataloader
    word_embedding_path = '/cs/student/msc/dsml/2023/mdavudov/UCB/ICD-MSMN/embedding/word2vec_sg0_100.model'

    PAD_CHAR = "**PAD**"
    PAD_INDEX = 150695

    dataset = MimicFullDataset(version, "test", word_embedding_path, 4000, summarised=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=False, num_workers=1)

    model.calculate_label_hidden()

    batch = next(iter(dataloader))
    batch_gpu = tuple([x.to(device) for x in batch])

    input_word, word_mask = batch_gpu[0:2]

    label_embed = model.label_feats

    hidden = model.calculate_text_hidden(input_word, word_mask)

    label_hidden = model.calculate_text_hidden(model.c_input_word, model.c_word_mask)


    # Calculate cosine similarity
    cosine_similarities = calculate_cosine_similarity(hidden[0], label_embed)
    
    
    metric, (yhat, y, yhat_raw), threshold = eval_func_cosine(dataloader, device, tqdm_bar = True, args = model.args)

    print_metrics(metric, suffix='Test')
    print(f"Threshold: {threshold}")
    

    

    
