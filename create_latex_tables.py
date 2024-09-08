
import json
from data_util import MimicFullDataset, my_collate_fn, aweful_collate_function

#load metrics files
metrics_attn_rep = json.load(open('metrics_attn_rep.json'))
metrics_attn_unique = json.load(open('metrics_attn_unique.json'))
metrics_attn_all = json.load(open('metrics_attn_all.json'))

metrics_ig_rep = json.load(open('metrics_ig_rep.json'))
metrics_ig_unique = json.load(open('metrics_ig_unique.json'))
metrics_ig_all = json.load(open('metrics_ig_all.json'))


word_embedding_path = '/cs/student/msc/dsml/2023/mdavudov/UCB/ICD-MSMN/embedding/word2vec_sg0_100.model'


dataset = MimicFullDataset('mimic3-50', "test", word_embedding_path, 4000, summarised=False)

code_desc = {}
ind_desc = dataset.extract_label_desc(dataset.ind2c)

for ind, code in dataset.ind2c.items():
    code = dataset.ind2c[ind]
    code_desc[code] = ind_desc[ind]

print(code_desc)
code_desc["all"] = "Average across all codes"
metrics_table_header = """
\\begin{table}[ht!]
    \centering
    \\begin{tabular}{| c | c | c | c | c | c | c |}
    \hline
    & \multicolumn{3}{c|}{Attention} & \multicolumn{3}{c|}{Integrated Gradients} \\\\
    & top-k (rep) & top-k (unique) & all words &top-k (rep) & top-k (unique) & all words \\\\
    \hline
"""

metrics_table_unit = """
    & \multicolumn{{6}}{{c|}}{{ {code} - {desc} }} \\\\
    AUROC & {auroc_attn_rep} & {auroc_attn_un} & {auroc_attn_all} & {auroc_ig_rep} & {auroc_ig_un} & {auroc_ig_all} \\\\
    AP & {ap_attn_rep} & {ap_attn_un} & {ap_attn_all} & {ap_ig_rep} & {ap_ig_un} & {ap_ig_all} \\\\
    \hline
"""


metrics_table_footer ="""
    \end{tabular}
    \caption{Results from Experiment 3 (Section \\ref{corrcode}), demonstrating how well word importance score predicts presence is code description ($k = 5$).}
    \label{result_metric}
\end{table}
"""

#print(metrics_table_header + metrics_table_unit.format(code='132', desc='2332') + metrics_table_unit.format(code='132', desc='2332') + metrics_table_footer)


metric_table = metrics_table_header


#round to two decimal places

def myround(val):
    try:
        return round(val, 4)
    except:
        return val


acc_table_header = """
\\begin{longtable}{| c | c | c |}
        \hline
        & Attention & Integrated Gradients \\\\
        \hline
"""

acc_table_unit = """
        & \multicolumn{{2}}{{c|}}{{ {code} - {desc} }} \\\\
        top-k (repetition) & {acc_rep_attn}\% & {acc_rep_ig}\% \\\\
        \hline
        top-k (unique) & {acc_un_attn}\% & {acc_un_ig}\%\\\\
        \hline
        all words & \multicolumn{{2}}{{c|}}{{{acc_all}\%}}\\\\
        \hline
"""   

acc_table_footer = """
    
    \caption{Results from Experiment 2 (Section \\ref{directword}), showing percentage of top keywords, with respect to the importance score, matched to the code description ($k = 5$).}
    \label{word_matching}
\end{longtable}
"""
# for code in metrics_attn_rep:
    
#     metric_table += metrics_table_unit.format(code=code, desc=code_desc[code],
#                                               auroc_attn_rep=myround(metrics_attn_rep[code]['auroc']),
#                                                 auroc_attn_un=myround(metrics_attn_unique[code]['auroc']),
#                                                 auroc_attn_all=myround(metrics_attn_all[code]['auroc']),
#                                                 auroc_ig_rep=myround(metrics_ig_rep[code]['auroc']),
#                                                 auroc_ig_un=myround(metrics_ig_unique[code]['auroc']),
#                                                 auroc_ig_all=myround(metrics_ig_all[code]['auroc']),
#                                                 ap_attn_rep=myround(metrics_attn_rep[code]['auprc']),
#                                                 ap_attn_un=myround(metrics_attn_unique[code]['auprc']),
#                                                 ap_attn_all=myround(metrics_attn_all[code]['auprc']),
#                                                 ap_ig_rep=myround(metrics_ig_rep[code]['auprc']),
#                                                 ap_ig_un=myround(metrics_ig_unique[code]['auprc']),
#                                                 ap_ig_all=myround(metrics_ig_all[code]['auprc'])
#                                                 )
    
# metric_table += metrics_table_footer

mypercent = lambda x: round(x*100, 2)

acc_table = acc_table_header
# for code in metrics_attn_rep:
        
#         acc_table += acc_table_unit.format(code=code, desc=code_desc[code],
#                                                 acc_rep_attn=mypercent(metrics_attn_rep[code]['accuracy']),
#                                                 acc_rep_ig=mypercent(metrics_ig_rep[code]['accuracy']),
#                                                 acc_un_attn=mypercent(metrics_attn_unique[code]['accuracy']),
#                                                 acc_un_ig=mypercent(metrics_ig_unique[code]['accuracy']),
#                                                 acc_all=mypercent(metrics_attn_all[code]['accuracy'])

                                                    
#                                                     )
# acc_table += acc_table_footer

for code in code_desc:
    print(f"{code} - {code_desc[code]}\\\\")
#print(metric_table)