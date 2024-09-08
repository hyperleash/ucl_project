import torch
from torch.utils.data import DataLoader
from data_util import MimicFullDataset, my_collate_fn
from torchinfo import summary
import logging
import argparse
from captum.attr import IntegratedGradients, LayerIntegratedGradients, TokenReferenceBase
#from captum.attr import visualization
from visualisation_local import VisualizationDataRecord
from visualisation_local import visualize_text

import json
import numpy as np
from IPython.display import HTML

logger = logging.getLogger("explain")

class ICDModelInterpreter:
    
    def __init__(self, model, device, dataset):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.PAD_INDEX = 150695
        self.PAD_CHAR = "**PAD**"
    
    def get_words_from_ids(self, id_list, word_to_id_dict):
        id_to_word_dict = {v: k for k, v in word_to_id_dict.items()}  # Reverse the dictionary
        return [id_to_word_dict[id] for id in id_list]
    
    # Define a function to extract context around a token
    def get_context_around_token(self, token_idx, tokens, word_to_id_dict, window_size=2):
        """Extracts a context window of words around a given token index."""
        start = max(0, token_idx - window_size)
        end = min(len(tokens), token_idx + window_size + 1)  # +1 to include the token itself
        return self.get_words_from_ids(tokens[start:end], word_to_id_dict=word_to_id_dict)
    
    def ig_attribute(self, input_word, word_mask, target, reference_indices=None):
        lig = LayerIntegratedGradients(self.model.predict_proba, self.model.encoder.word_encoder.word_embedding)
        
        if reference_indices is None:
            reference_indices = TokenReferenceBase(reference_token_idx = self.PAD_INDEX).generate_reference(4000, device=self.device).unsqueeze(0)
        
        self.model.train()
        attr, delta = lig.attribute(input_word, reference_indices, target=target.item(), return_convergence_delta=True, additional_forward_args=word_mask)
        self.model.eval()

        return attr, delta

    def add_attributions_to_visualizer(self, attributions, text, pred, delta, vis_data_records, target, yhat, y, title = None, attn = False):
        #print(f"############ ATTRIBUTIONS: {attributions.shape}")
        #print(f"############ ATTRIBUTIONS: {attributions}")
        if not attn:
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions = attributions / torch.norm(attributions)
    
        else:
            #scale attention scores from to range 0-1 using z-score normalization
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

        attributions = attributions.cpu().detach().numpy()

        vis_data_records.append(VisualizationDataRecord(
                                attributions,
                                pred[0][target],
                                yhat[0][target],
                                y[0][target],
                                "",
                                attributions.sum(),
                                text,
                                delta,
                                )) 

    def get_prediction(self, batch_gpu):
        
        outputs = []
        with torch.no_grad():
            self.model.calculate_label_hidden()
                
            now_res = self.model.predict(batch_gpu, None)
            
            outputs.append({key:value.cpu().detach() for key, value in now_res.items()})
            
        yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy()
        yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
        y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

        return yhat, yhat_raw, y
    
    def get_attention_scores(self, batch):
        batch_gpu = tuple([x.to(self.device) for x in batch])
        input_word, word_mask = batch_gpu[0:2]
    
        yhat, yhat_raw, y = self.get_prediction(batch_gpu)

        tokens = batch[0][0].numpy()

        text = self.get_words_from_ids(tokens, self.dataset.word2id)

        idx = np.where(yhat[0] == 1)[0]
        
        codes = [self.model.decoder.ind2c[i] for i, x in enumerate(yhat[0]) if x]

        vis_data_records_ig_arr = {}

        for code_idx in idx:
        # Get the attention weights for this label and all heads
            label_alphas = self.model.decoder.attention_weights[0, code_idx, :, :] 
            
            #normalize attention scores
            normalized_alphas = torch.softmax(torch.from_numpy(label_alphas), dim=-1)

            #logger.debug(f"ICD-9 code: {model.decoder.ind2c[code_idx]}")
            for head_idx in range(normalized_alphas.shape[-1]):
                # Get the top tokens for this head and label
                _, top_indices = torch.topk(normalized_alphas[:, head_idx], k=10, sorted=True) 

                head_tokens = [tokens[i] for i in top_indices]
            
            aggregated_alphas, _ = torch.max(normalized_alphas, dim=-1)

            probs = self.model.predict_proba(batch_gpu[0], batch_gpu[1])

            code = self.model.decoder.ind2c[code_idx]
            logger.debug(f"{code_idx}: {code}")
            vis_data_records_ig = []

            #remove masked tokens from the text:
            masked_tokens = ["**UNK**", "**PAD**", "**MASK**"]
            text = [word for word in text if word not in masked_tokens]
            
            
            self.add_attributions_to_visualizer(aggregated_alphas, text, probs, [], vis_data_records_ig, code_idx, yhat, y, attn = True)
            vis_data_records_ig_arr[code] = vis_data_records_ig
        return vis_data_records_ig_arr
    
    def interpret(self, batch, title=None):
        tokens = batch[0][0].numpy()
        batch_gpu = tuple([x.to(self.device) for x in batch])
        input_word, word_mask = batch_gpu[0:2]
        
        yhat, yhat_raw, y = self.get_prediction(batch_gpu)

        text = self.get_words_from_ids(tokens, self.dataset.word2id)

        idx = np.where(yhat[0] == 1)[0]
        
        codes = [self.model.decoder.ind2c[i] for i, x in enumerate(yhat[0]) if x]

        probs = self.model.predict_proba(batch_gpu[0], batch_gpu[1])
        
        vis_data_records_ig_arr = {}
        for target in idx:
            code = self.model.decoder.ind2c[target]
            logger.debug(f"{target}: {code}")
            vis_data_records_ig = []
            attr, delta = self.ig_attribute(input_word, word_mask, target)
            self.add_attributions_to_visualizer(attr, text, probs, delta, vis_data_records_ig, target, yhat, y, title)
            vis_data_records_ig_arr[code] = vis_data_records_ig
        return vis_data_records_ig_arr
    
    def visualise(self, vis_data_records_arr, file_name = "visualization.html"):
        for key in vis_data_records_arr:
            html_vis = visualize_text(vis_data_records_arr[key])
            html_str = html_vis.data

            with open(f"visualization_{key}.html", "w") as f:
                f.write(html_str)

    def visualize_codes(self, datarecords_dict_ig: dict[str, VisualizationDataRecord], datarecords_dict_attn: dict[str, VisualizationDataRecord], code_desc, file_name = "visualisation.html", legend: bool = True) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
        
        dom = ['<div class="visualization-container">']  # Main container for all visualizations

        # Dropdown to select ICD code
        dom.append('<select id="icd-code-select" onchange="updateVisualization()">')
        for code in datarecords_dict_ig.keys():
            dom.append(f'<option value="{code}">{code} - {code_desc[code]}</option>')
        dom.append('</select>')
        dom.append("<br>")
        # Dropdown for method selection
        dom.append('<select id="method-select" onchange="updateVisualization()">')
        dom.append('<option value="ig">Integrated Gradients</option>')
        dom.append('<option value="attn">Attention</option>')
        dom.append('</select>')
        dom.append('<br>')
            
        # Initial visualization (first ICD code)
        initial_code = list(datarecords_dict_ig.keys())[0]
        dom.append(f'<div id="visualization-{initial_code}" class="visualization">')
        dom.append("</div>")

        for code in datarecords_dict_ig.keys():
            dom.append(f'<div id="visualization-{code}-ig" class="visualization" style="display: none">')
            dom.append(visualize_text(datarecords_dict_ig[code]).data)
            dom.append('</div>')

            # Attention method container
            dom.append(f'<div id="visualization-{code}-attn" class="visualization" style="display: none;">')
            dom.append(visualize_text(datarecords_dict_attn[code]).data)  
            dom.append('</div>')


        # dom.append(create_visualization_table(datarecords_dict[initial_code], legend))  # Create a table for visualization
        # dom.append('</div>')

        # JavaScript to update visualization
        dom.append("""
        <script>
            function updateVisualization() {
                const selectedCode = document.getElementById('icd-code-select').value;
                const selectedMethod = document.getElementById('method-select').value;
                const visualizations = document.querySelectorAll('.visualization');

                visualizations.forEach(vis => {
                    if (vis.id === `visualization-${selectedCode}-${selectedMethod}`) {
                        vis.style.display = 'block';
                    } else {
                        vis.style.display = 'none';
                    }
                });
            }

            

        </script>
        """)
        dom.append('</div>')  # Close main container
        html = HTML("".join(dom))
        html_str = html.data
        with open(f"{file_name}", "w") as f:
                f.write(html_str)

    


if __name__ == "__main__":

    #Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=bool, default=True)
    parser.add_argument('-a', '--algorithm', type=str, default='attn')
    
    args = parser.parse_args()

    #Setting up logger
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    current_log_level = log_levels[min(len(log_levels) - 1, args.verbose)]
    logging.basicConfig(level=current_log_level)
    logger = logging.getLogger("explain")

    if args.model_path.find('mimic3-50') >= 0:
        version = 'mimic3-50'
    else:
        version = 'mimic3'

    logger.info(f"Version: {version}")
    logger.info(f"Mechanism: {args.algorithm}")
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

    dataset = MimicFullDataset(version, "test", word_embedding_path, 4000, summarised=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn, shuffle=False, num_workers=1)

    #Initialising interpreter
    interpreter = ICDModelInterpreter(model, device, dataset)

    code_desc = {}
    ind_desc = dataset.extract_label_desc(dataset.ind2c)

    for ind, code in dataset.ind2c.items():
        code = dataset.ind2c[ind]
        code_desc[code] = ind_desc[ind]

    note = next(iter(dataloader))

    vis_data_records_ig = interpreter.interpret(note)

    vis_data_records_attn = interpreter.get_attention_scores(note)

    interpreter.visualize_codes(vis_data_records_ig, vis_data_records_attn, code_desc, "combined.html")



