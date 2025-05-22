# BLUR: A Bi-Level Optimization Approach for LLM Unlearning


## 🚀 Abstract 

Enabling large language models (LLMs) to unlearn knowledge and capabilities acquired during training has proven vital for ensuring compliance with data regulations and promoting ethical practices in generative AI. Although there are growing interests in developing various unlearning algorithms, it remains unclear how to best formulate the unlearning problem. The most popular formulation uses a weighted sum of forget and retain loss, but it often leads to performance degradation due to the inherent trade-off between forget and retain losses. In this work, we argue that it is important to model the hierarchical structure of the unlearning problem, where the forget problem (which *unlearns* certain knowledge and/or capabilities) takes priority over the retain problem (which preserves model utility). This hierarchical structure naturally leads to a bi-level optimization formulation where the lower-level objective focuses on minimizing the forget loss, while the upper-level objective aims to maintain the model's utility. Based on this new formulation, we propose a novel algorithm, termed Bi-Level UnleaRning ($\texttt{BLUR}$), which not only possesses strong theoretical guarantees but more importantly, delivers superior performance. In particular, our extensive experiments demonstrate that $\texttt{BLUR}$ consistently outperforms all the state-of-the-art algorithms across various unlearning tasks, models, and metrics.

## ⚙️ Installation  
### **MUSE Environment** 
To set up the MUSE environment, run:
```sh
conda env create -f environment.yml
conda activate muse_env
```

### **WMDP Environment**  
For the WMDP environment, follow these steps:
```sh
conda create -n wmdp_env python=3.9.21 -y
conda activate wmdp_env
conda install pytorch=2.1.1 torchvision=0.16.1 torchaudio=2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install datasets==3.2.0 wandb==0.19.2 transformers==4.37.2 sentencepiece==0.1.99 sentence-transformers==2.5.1
pip install terminaltables==3.1.10 sacrebleu==2.4.0 rouge-score==0.1.2 matplotlib==3.8.3 seaborn==0.13.2 scikit-learn==1.4.0
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```
---

## 📌 Running the Experiments  

### **Running MUSE**
To execute the MUSE experiments, run the following script:  
```sh
CORPUS="news"
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR="muse-bench/MUSE-News_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"
MAX_LEN=2048
EPOCHS=10
LR='2.5e-5'
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs
GAMA=1.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for algo in 'BLO_forget_lower_npo_gdr'; do
        python unlearn.py \
            --algo $algo \
            --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
            --data_file $FORGET --retain_data_file $RETAIN \
            --out_dir "/home/mhong/shared/hadir/out_dir/$CORPUS/$algo" \
            --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
            --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
            --gama $GAMA 
done
```
---

### **Running WMDP**  

To execute the WMDP experiments, use the following command:  

```sh
CUDA_VISIBLE_DEVICES=0 python3 -m rmu.unlearn_bi \
    --max_num_batches 150 --batch_size=4 \
    --retain_corpora wikitext,wikitext \
    --forget_corpora bio_remove_dataset,cyber-forget-corpus \
    --steering_coeffs 6.5,6.5 --alpha 800,800 \
    --lr 5e-5 --seed 0 --output_dir models/bi_unlearn
```

---

## 🐝 Citation  

```bibtex
@article{BLUR2025,
  author    = {Hadi Reisizadeh, Jinghan Jia, Zhiqi Bu,  Bhanukiran Vinzamuri, Anil Ramakrishna, Kai-Wei Chang, Volkan Cevher, Sijia Liu, Mingyi Hong},
  title     = {BLUR: A Bi-Level Optimization Approach for LLM Unlearning},
  journal   = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/abs/...}
}
```
