import os
import subprocess
import pdb

""" 
1. Estrazione feature CPTAC
2. Creazione splits
2. Train mutazioni su ./splits/task_3_tumor_mutation_100_KRAS

default early stop and CE loss
"""
Split = False
Train = False
Eval = True
heatmaps = False
os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libffi.so.7"


#genes = ['KRAS', 'TP53', 'SMAD4', 'CDKN2A']


genes = ['TP53']
#f_extractors = ['UNI_wsi_level', 'Resnet', 'CONCH']
f_extractors = ['CONCH']

model_size = ['big']




for gene in genes:

    if Split:
        print(f'Running Splitting Command for {gene}')
        subprocess.run(["/home/berloco/anaconda3/envs/clam_uni/bin/python", 'create_splits_seq.py', '--task',  'task_3_HE_vs_mut', '--seed', '1',
                        '--label_frac', '1',  '--k', '10', '--target', '{}'.format(gene), '--csv_path', './dataset_csv/cptac_csv', '--target', gene])

    if Train:
        for f_extractor in f_extractors:
            for size in model_size:
                print(f"Training CLAM for {gene}, with CLAM {size} model using features extracted by {f_extractor}")
                pdb.run(["/home/berloco/anaconda3/envs/clam_uni/bin/python", "main.py", "--drop_out",  "--lr", "1e-5",  "--weighted_sample", "--bag_loss", "ce",
                            "--task", "task_3_HE_vs_mut", "--model_type", "clam_sb", "--data_root_dir", "/mnt/d/Datasets/Pancreas_Features",
                             "--split_dir", "tcga/task_3_HE_vs_mut_100_{}".format(gene), "--exp_code", "train_tcga_x20_{}_{}".format(gene, f_extractor),
                             "--early_stopping", "--csv_path", "./dataset_csv/cptac_csv", "--model_size", size, "--results_dir", "./results/mut_TCGA_model_{}_{}".format(size, f_extractor),
                                '--target', gene, '--feat_extractor', f_extractor],
                            env={'CUDA_VISIBLE_DEVICES': '0'}
                            )


    if Eval:
        for f_extractor in f_extractors:
            for size in model_size:
                print("Running Inference!")
                print(f"Evaluating TCGA on CPTAC for {gene}, with CLAM model with features extractors {f_extractor}")
                subprocess.run(["/home/berloco/anaconda3/envs/clam_uni/bin/python", "eval.py",  "--data_root_dir", "/mnt/d/Datasets/Pancreas_Features",  "--results_dir", "./results/mut_TCGA_model_{}_{}".format(size, f_extractor),
                                "--models_exp_code", "train_tcga_x20_{}_{}_s1".format(gene, f_extractor),
                                "--save_exp_code", "task_MUT_tcga_{}_{}".format(gene, size), "--task",  "task_3_HE_vs_mut", "--csv_path", "{}".format(gene),
                                "--model_size", size, "--split", "all", '--feat_extractor', f_extractor],
                               env={'CUDA_VISIBLE_DEVICES': '0'}
                               )


    if heatmaps:
        print(f"Creating Heatmaps for gene: {gene}!")
        subprocess.run(["/home/berloco/anaconda3/envs/clam_uni/bin/python", "create_heatmaps.py", "--config", "config_{}.yaml".format(gene)],
                       env={'CUDA_VISIBLE_DEVICES': '0'}
                       )



