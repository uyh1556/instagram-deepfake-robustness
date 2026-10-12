# Instagram Deepfake Robustness

Research workspace for deepfake detection under Instagram-like image filters using FaceForensics++ raw/c0 data.

## Local environment

- Python: 3.12.8
- Virtual environment: `.venv`
- Package versions: `requirements-local.txt`, captured from the existing `deepfake_project` environment

```bash
source .venv/bin/activate
python -m pip install -r requirements-local.txt
```

The local dependency snapshot reproduces the existing macOS environment. A separate dependency file will be prepared for Colab GPU experiments when needed.

Datasets, checkpoints, outputs, and the virtual environment are excluded from Git. Existing datasets and checkpoints remain in their original locations.

## Experiment status

Repository and local environment setup only. Filter selection and the augmentation/domain-adaptation experiment protocol have not yet been frozen.
