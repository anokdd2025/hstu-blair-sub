### Integrating Textual Embeddings from Contrastive Learning with a Generative Recommender for Enhanced Personalization

We present **HSTU-BLaIR**, a hybrid framework that augments the Hierarchical Sequential Transduction Unit (HSTU) generative recommender with BLaIR—a contrastive text embedding model. This integration enriches item representations with semantic signals from textual metadata while preserving HSTU's powerful sequence modeling capabilities.

We evaluate our method on two domains from the Amazon Reviews 2023 dataset, comparing it against the original HSTU and a variant that incorporates embeddings from OpenAI’s state-of-the-art `text-embedding-3-large` model. While the OpenAI embedding model is likely trained on a substantially larger corpus with significantly more parameters, our lightweight BLaIR-enhanced approach—pretrained on domain-specific data—consistently achieves better performance, highlighting the effectiveness of contrastive text embeddings in compute-efficient settings.


---

## 🚀 Getting Started

Install the required Python packages with ```pip3 install -r requirements.txt```.

**HSTU-BLaIR** is built on top of [the HSTU-based generative recommender (commit `ece916f`)](https://github.com/facebookresearch/generative-recommenders/tree/ece916f) and extends it with additional modules for integrating textual embeddings. It has been tested on Ubuntu 22.04 with Python 3.9, CUDA 12.6, and a single NVIDIA RTX 4090 GPU.

---

## 🧪 Experiments

To reproduce the experiments on the Amazon Reviews 2023 dataset, follow these steps:

### 1. Download and preprocess the data

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

To use OpenAI `text-embedding-3-large` embeddings instead of BLaIR, update the corresponding line in `preprocess_public_data.py` as follows:

```python
text_embedding_model = "blair"
```

to

```python
text_embedding_model = "openai"
```

Make sure you have correctly set up your OpenAI API credentials if you choose this option.


### 2. Run the model

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/amzn23_game/hstu-sampled-softmax-n512-blair.gin --master_port=12345
```

You can find other configuration files in the configs/ directory for different settings and domains.

