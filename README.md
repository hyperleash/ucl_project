# ICD-MSMN
Code for the UCL final project as part of MSc Data Science and Machine Learning. Code is extended from "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic ICD Coding" [ACL 2022] - https://github.com/GanjinZero/ICD-MSMN

# Environment


# Dataset
In order to gain access to the MIMIC-III dataset, you first need to obtain a licence from PhysioNet: https://physionet.org/content/mimiciii/1.4/

Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset.
You should obtain **train_full.csv**, **test_full.csv**, **dev_full.csv**, **train_50.csv**, **test_50.csv**, **dev_50.csv** after preprocessing.
Please put them under **sample_data/mimic3**.
Then you should use **preprocess/generate_data_new.ipynb** for generating json format dataset.

# Word embedding
Please download [word2vec_sg0_100.model](https://github.com/aehrc/LAAT/blob/master/data/embeddings/word2vec_sg0_100.model) from LAAT.
You need to change the path of word embedding.

# Training
To train a model used in this project, run 1GPU_run_50.sh:

```
sh ./1GPU_run_50.sh
```

# Evaluate checkpoints
Once a model is trained, it can be evaluated like so:
```
python eval_model.py MODEL_CHECKPOINT
```


# Citation for the original MSMN paper
```
@inproceedings{yuan-etal-2022-code,
    title = "Code Synonyms Do Matter: Multiple Synonyms Matching Network for Automatic {ICD} Coding",
    author = "Yuan, Zheng  and
      Tan, Chuanqi  and
      Huang, Songfang",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.91",
    pages = "808--814",
    abstract = "Automatic ICD coding is defined as assigning disease codes to electronic medical records (EMRs).Existing methods usually apply label attention with code representations to match related text snippets.Unlike these works that model the label with the code hierarchy or description, we argue that the code synonyms can provide more comprehensive knowledge based on the observation that the code expressions in EMRs vary from their descriptions in ICD. By aligning codes to concepts in UMLS, we collect synonyms of every code. Then, we propose a multiple synonyms matching network to leverage synonyms for better code representation learning, and finally help the code classification. Experiments on the MIMIC-III dataset show that our proposed method outperforms previous state-of-the-art methods.",
}
```
