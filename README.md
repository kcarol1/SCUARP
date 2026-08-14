## Environment

Install a PyTorch build that matches the CUDA version on the machine, then
install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This release was sanity-checked with Python 3.9.21, PyTorch 2.5.1, NumPy 2.0.2,
SciPy 1.13.1, and scikit-learn 1.6.0.

## Running the paper demo

`demo.py` intentionally keeps the original direct-run workflow. There is no new
command-line interface: select one dataset by editing the configuration blocks
near the top of the file, keep only one block active, and run the script from
the repository root:

```bash
python demo.py
```

## Citation
If you find this work useful, please consider citing:
```bibtex
@article{Xiao2026SCUARP,
  author    = {Deng, Juan and Xiao, Luxi and Liu, Shujun},
  title     = {Enhanced Multi-view Subspace Clustering via Unified Anchor Graph and Random Projection for Hyperspectral and LiDAR Data},
  journal   = {IEEE Access},
  year      = {2026},
  doi       = {10.1109/ACCESS.2026.3724248},
  publisher = {IEEE}
}
```