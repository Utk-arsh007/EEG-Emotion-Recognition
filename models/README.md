# Models and Checkpoints Directory

This directory stores trained deep learning model weights and checkpoints for:
- `checkpoints/best_hybrid_torch.pt`: Proposed Hybrid CNN-BiLSTM-Attention PyTorch weights.
- `checkpoints/best_cnn_torch.pt`: Model A (CNN) PyTorch weights.
- `checkpoints/best_bilstm_torch.pt`: Model B (BiLSTM) PyTorch weights.
- `checkpoints/best_ablation_torch.pt`: Ablation (Hybrid without Attention) PyTorch weights.
- `checkpoints/*.keras`: Keras format model weights generated during Google Colab T4 GPU training.

### Loading Checkpoints in Code:
```python
from src.predict import EEGPredictor
predictor = EEGPredictor(checkpoint_path="./models/checkpoints/best_hybrid_torch.pt")
result = predictor.predict_from_file("./sample_data/s01.dat", trial_idx=0)
print(result["predictions"])
```
