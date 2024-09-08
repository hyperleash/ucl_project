from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoConfig
import torch
import json
import pandas as pd
from tqdm import tqdm
import argparse
from summarizer import Summarizer

parser = argparse.ArgumentParser(description='Summarisation script for MIMIC-III dataset')
parser.add_argument('--type', type=str, default="abstractive", help='Type of summarization model (abstractive or extractive)')
parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size for the summarization model')
parser.add_argument('--overlap', type=int, default=50, help='Overlap between chunks for the summarization model')
parser.add_argument('--input_file', type=str, help='Path to the input file with notes', required=True)
parser.add_argument('--output_file', type=str, help='Path to the output file with summaries', default="")
parser.add_argument('--addition', type=bool, default=False, help='Use additional information for summarization')

args = parser.parse_args()

if(args.type == "abstractive"):
    tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")

    model = model.to("cuda")
    config = model.config

elif(args.type == "extractive"):
    config = AutoConfig.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    config.output_hidden_states=True
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    custom_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', config=config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=tokenizer)
else:
    raise ValueError("Invalid type of summarization model")

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

def summarize_note_chunked(note_text, chunk_size=512, overlap=50):
    #print("Starting tokenization on the note text of length", len(note_text))
    inputs = tokenizer(note_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    #print("Tokenization complete")
    # Chunk creation with overlap
    chunks = []
    start_idx = 0
    while start_idx < input_ids.size(1):
        end_idx = min(start_idx + chunk_size, input_ids.size(1))
        chunks.append(input_ids[:, start_idx:end_idx])
        start_idx += (chunk_size - overlap)

    # Summarization loop
    summary_parts = []
    for chunk in chunks:
        #print(chunk.shape)
        chunk_summary_ids = model.generate(
            chunk,
            max_length=8192,  
            num_beams=4,
            early_stopping=True,
        )
        summary_parts.append(tokenizer.decode(chunk_summary_ids[0], skip_special_tokens=True))

    # Combine summaries
    final_summary = " ".join(summary_parts)
    return final_summary

def summarize_extractive(note_text):
    return model(note_text)

file_path = args.input_file

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

summaries = []

df_notes = df['TEXT']

if(args.output_file == ""):
    output_file_path = args.input_file

if(args.addition):
    print("Using additional information for summarization")
print(f"Summarizing {len(df)} notes")
print(f"Using chunks of {args.chunk_size} tokens with an overlap of {args.overlap} tokens")
print(f"Input file: {file_path}")
print(f"Output file: {output_file_path}")

tqdm.pandas()
# Iterate over each row and apply the summarization function
for i, note_text in tqdm(enumerate(df['TEXT']), desc="Summarizing Notes", total=len(df)): 

    if(args.addition):
        additional_info = df['Addition'][i]
        words = additional_info.split()
        if len(words) > 1000:
            additional_info = " ".join(words[:1000])

        note_text = note_text + " " + additional_info
    # Check if note_text is a string (to avoid errors if there are non-string values)
    if isinstance(note_text, str):
        if(args.type == "abstractive"):
            summary = summarize_note_chunked(note_text, args.chunk_size, args.overlap)
        elif(args.type == "extractive"):
            summary = summarize_extractive(note_text)
        summaries.append(summary)
    else:
        print(f"Skipping non-string input")
        summaries.append(None)  # Or some other default value for non-string inputs

if(args.type == "extractive"):
    new_column_name = f"summary_extractive"
elif(args.type == "abstractive"):
    new_column_name = f"summary_chunk_{args.overlap}"

if(args.addition):
    new_column_name += "_addition"

df[new_column_name] = summaries


df.to_json(output_file_path, orient="records")

print(f"DataFrame with summaries saved to {output_file_path}")