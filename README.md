# EvoRGCN: Evolutionary-Scale RGCN for Protein–Protein Interaction Prediction

This repository contains the code, notebooks, pretrained models, and data processing pipelines for **EvoRGCN**, a hybrid framework that enhances Relational Graph Convolutional Networks (RGCN) with protein language model embeddings (ProtBERT & ESM-2) to deliver state‑of‑the‑art performance on protein–protein interaction (PPI) prediction tasks.

📄 **Full project report**: `EvoRGCN_EvoRGCN at main · MohKu311/EvoRGCN.pdf`

---

## 📂 Repository Structure

```
EvoRGCN/
├── sh27k_gcn.py                   # Script: mode-specific GCN on SH27K
├── protein_to_seq.pkl             # Pickled mapping: protein → raw sequence
├── esm2_embeddings_linkprediction_rgcn.pkl  # Cached ESM-2 embeddings
├── protbert_embeddings_linkprediction.pkl   # Cached ProtBERT embeddings (SH27K)
├── protbert_embeddings_linkprediction_148k.pkl  # Cached ProtBERT embeddings (SH148K)
├── gcn_link_prediction_*.pth           # Pretrained GCN model checkpoints
├── gat_link_prediction_*.pth           # Pretrained GAT model checkpoints
├── protbert_rgcn_link_predictor_*.pth  # ProtBERT+RGCN checkpoints
├── rgcn_link_prediction_model_*.pth    # One-Hot+RGCN checkpoints
├── rgcn_esm2_linkpredictor_*.pth       # ESM-2+RGCN checkpoints
├── SH27k-preprocess.ipynb         # Notebook: preprocess SH27K subset
├── SH148k-preprocess.ipynb        # Notebook: preprocess SH148K subset
├── SH27k-graph.ipynb              # Notebook: build & explore SH27K graph
├── esm2_27k.ipynb                  # Notebook: generate ESM-2 embeddings on SH27K
├── esm2_27k-directional.ipynb      # Notebook: ESM-2 embeddings + directed SH27K
├── esm2_148k.ipynb                 # Notebook: generate ESM-2 embeddings on SH148K
├── esm2_148k-directional.ipynb     # Notebook: ESM-2 embeddings + directed SH148K
├── gcn_link_prediction_*.ipynb     # GCN link-prediction notebooks
├── gat_link_prediction_*.ipynb     # GAT link-prediction notebooks
├── protbert_*.ipynb                # ProtBERT+RGCN link-prediction notebooks
├── rgcn_*.ipynb                    # RGCN link-prediction notebooks (One-Hot & ESM-2)
├── protein-links-visualize.ipynb   # Notebook: visualize PPI subgraphs in STRING style
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/MohKu311/EvoRGCN.git
cd EvoRGCN/EvoRGCN
```

### 2. Install Dependencies
EvoRGCN relies on the following Python packages:
- torch
- dgl (or torch-geometric)
- transformers
- scikit-learn
- pandas
- numpy

To install them, run:
```bash
pip install torch dgl torch-geometric transformers scikit-learn pandas numpy
```

### 3. Download and preprocess data
- **STRING v11.0**: download Homo sapiens PPI TSV from [STRING database](https://string-db.org).
- **Run** the preprocessing notebooks:
  - `SH27k-preprocess.ipynb` to create the balanced SH27K dataset
  - `SH148k-preprocess.ipynb` to create the balanced SH148K dataset

### 4. Generate sequence embeddings (optional)
- **ProtBERT**: run `protbert_embeddings_linkprediction_*.ipynb`
- **ESM-2 esm2_t6_8M_UR50D**: run `esm2_27k.ipynb` & `esm2_148k.ipynb`

### 5. Train or evaluate models
- **GCN**:
```bash
python sh27k_gcn.py --dataset SH27K --mode activation   # example for activation mode
```
- **GAT & RGCN**: see the corresponding notebooks (`gat_link_prediction_*.ipynb`, `rgcn_*.ipynb`)

### 6. Visualize
Use `protein-links-visualize.ipynb` to render PPI subgraphs in an interactive STRING‑style layout.

---

## 📊 Results & Report
- All ROC-AUC and PR-AUC tables, plots, and detailed analysis are in the project report (PDF) and the `*.ipynb` notebooks under `EvoRGCN/EvoRGCN`.

---

## 🤝 Contributing
Contributions are welcome via GitHub issues or pull requests. Please follow the existing code style and include tests if possible.
