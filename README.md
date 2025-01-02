# StructFormer: Document Structure-based Masked Attention and its Impact on Language Model Pre-Training

This repository contains the official implementation of "StructFormer: Document Structure-based Masked Attention and its Impact on Language Model Pre-Training", which introduces a novel approach for incorporating document structure awareness in transformer models through masked attention during pre-training.

## Overview

Traditional transformer models have been limited by their computational requirements when handling long documents. StructFormer introduces:

- Structure-aware pre-training using document headers as global tokens
- Efficient sparse attention mechanism based on Longformer
- Model-agnostic approach that works with various downstream tasks
- Single-round communication with significant computational efficiency
- Formal privacy guarantees for embedding sharing

## Repository Structure

```
├── scratch_train/          # Pre-training implementation
│   ├── config.json         # Configuration file
│   ├── op_on_pkl_files.py  # Pickle file operations
│   ├── sepTestTrain.py     # Test/train separation utilities
│   ├── split_dataset.py    # Dataset splitting utilities
│   ├── train_mlm.py        # Masked language model training
│   ├── train_mlm_no_pkl.py # MLM training without pickled data
│   ├── trainer.py          # Main trainer implementation
│   └── trainer_hf.py       # HuggingFace trainer utilities
├── SF_test/                # Evaluation and testing code
│   ├── GLUE-baselines/     # GLUE benchmark evaluation
│   └── models/             # Model implementations
├── Test/                   # Additional test files
└── .gitignore             # Git ignore file
```

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Datasets
- Additional requirements will be listed in requirements.txt

## Key Features

1. **Structure-Aware Pre-training**:
   - Uses document headers as global attention tokens 
   - Masked Language Modeling with structural context
   - Efficient sparse attention mechanism

2. **Evaluation Suite**:
   - SciREX benchmark evaluation
   - GLUE benchmark testing
   - Attention pattern analysis

## Usage

### Pre-training

```python
# Configure pre-training settings
python scratch_train/train_mlm.py \
    --config config.json \
    --model_path path/to/save \
    --data_path path/to/data

# Run evaluation
python SF_test/evaluate.py \
    --model_path path/to/model \
    --task scirex
```

### Attention Analysis

```python
# Analyze attention patterns
python SF_test/analyze_attention.py \
    --model_path path/to/model \
    --test_data path/to/test_data
```

## Results 

### SciREX Results
| Task | Precision | Recall | F1 |
|------|-----------|--------|----| 
| Salient Clusters | 0.2581 | 0.6127 | 0.3419 |
| Binary Relations | 0.0550 | 0.5100 | 0.0890 |
| 4-ary Relations | 0.0019 | 0.2760 | 0.0037 |

### GLUE Benchmark Results
| Task | Metric | Score |
|------|--------|-------|
| CoLA | Mathews Correlation | 0.469 |
| STSB | Combined Score | 0.856 |
| MRPC | F1 | 0.927 |
| QNLI | Accuracy | 0.910 |
| SST2 | Accuracy | 0.933 |

## Contribution Guidelines

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Citation

```bibtex
@article{structformer2024,
  title={StructFormer: Document Structure-based Masked Attention and its Impact on Language Model Pre-Training},
  author={Ponkshe, Kaustubh and Subramanian, Venkatapathy and Modani, Natwar and Ramakrishnan, Ganesh},
  year={2024}
}
```

## License

[Add License Information]

## Contact

For questions and feedback, please create an issue in the repository or contact the authors.

## Acknowledgments

We thank the authors of Longformer and other baseline models used in our comparisons.