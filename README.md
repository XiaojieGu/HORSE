# [Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models](https://arxiv.org/pdf/2601.11441) [![arXiv](https://img.shields.io/badge/arXiv-2601.11441-b31b1b.svg?style=plastic)](https://arxiv.org/pdf/2601.11441)


## 📦 Data Preparation

Download the datasets from [Google Drive](https://TBD) and place them under:

```
data/
├── zsre/
│   ├── zsre_train.json
│   └── zsre_eval.json
└── counterfact/
    ├── cf_train.json
    └── cf_eval.json
```

## 🚀 Setup

```bash
conda create -n horse python=3.10
conda activate horse
pip install -r requirements.txt
```

## 🧪 Run

```bash
sh run.sh
```

### Configuration

Modify `run.sh` to customize experiments:

```bash
python main.py data=zsre \
    model=llama2-7b \
    editor=horse \
    editor.lr=1e-5 \
"

**Supported Options:**

| Parameter | Options |
**Key Hyperparameters:**
## 📁 Project Structure

```
├── config/
│   ├── config.yaml
│   ├── data/          # Dataset configs
│   ├── editor/        # Editor configs (horse, malmen, mend)
│   └── model/         # Model configs
├── data/              # Dataset files & loaders
├── editor/            # Editor implementations
├── nets.py            # MALMENNet architecture
├── util.py            # Utility functions
└── main.py            # Entry point
```

## 🙏 Acknowledgements

Our work is based on [MALMEN](https://github.com/ChenmienTan/malmen).

## 📫 Contact

For any inquiries, feel free to reach out at **peettherapynoys@gmail.com**

## 📑 Citation
If you find **HORSE** useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{gu2026hierarchicalorthogonalresidualspread,
  title={Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models},
  author={Gu, Xiaojie and Chen, Guangxu and Yang, Yuheng and Han, Jingxin and Zhang, Andi},
  booktitle={ICASSP 2026 -- IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2026},
  organization={IEEE}
}
