# ether0.remotes

Server code for ether0 reward functions.

## Molecular Transformer (MolTrans) Model

To run the `/translate` endpoint, you need a pre-trained MolTrans PyTorch model.
This can be acquired from Future House's Google Drive via the following command:

```bash
curl --location --output src/ether0/USPTO480k_model_step_400000.pt \
  "https://drive.usercontent.google.com/download?id=1Rjd3wXg2oLeCpNUofFRvVvQoOcgWd6vf&export=download&confirm=t"
```

Or more manually:

1. Go to this notebook: https://github.com/schwallergroup/ai4chem_course/blob/main/notebooks/07%20-%20Reaction%20Prediction/template_free.ipynb
2. Download the `USPTO480k_model_step_400000.pt`
   linked in the `trained_model_url` variable's linked Google Drive file:
   https://drive.google.com/uc?id=1ywJCJHunoPTB5wr6KdZ8aLv7tMFMBHNy
3. Set the environment variable `ETHER0_REMOTES_MOLTRANS_MODEL_PATH`
   to the downloaded PyTorch model's location,
   or place the model in the default checked `ether0` source code folder (`src/ether0`).

## Serving

To run the server:

1. `pip install` with the `serve` extra: `pip install ether0.remotes[serve]`
2. Then run the following command:

```bash
ETHER0_REMOTES_API_TOKEN="abc123" \
ETHER0_REMOTES_MOLTRANS_MODEL_PATH="/path/to/downloaded/USPTO480k_model_step_400000.pt" \
ether0-serve
```
