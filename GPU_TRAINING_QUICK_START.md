# 🚀 GPU Training Guide - Quick Start

## ✅ You Now Have TWO Ways to Use GPU

### Option 1: Simple GPU Toggle (Original Cell - Modified)
In the **"Manual Approach"** cell, I added a simple `USE_GPU` toggle:

```python
# 🔧 GPU TOGGLE: Set to True to enable GPU for XGBoost/LightGBM
USE_GPU = True  # Change to False to use CPU only
```

**To Use:**
1. Set `USE_GPU = True` (already set)
2. Run the cell
3. XGBoost and LightGBM will automatically use GPU if available

### Option 2: Full GPU Training (New Cell - Comprehensive)
In the new **"Manual Approach with GPU Acceleration"** cell, you get:
- ✅ Automatic GPU detection
- ✅ XGBoost GPU & CPU comparison
- ✅ LightGBM GPU & CPU comparison  
- ✅ CuML GPU models (if installed)
- ✅ Training time tracking
- ✅ GPU vs CPU speedup statistics

**To Use:**
1. Run all the cells in the "GPU Acceleration" section
2. Watch for GPU utilization messages
3. Compare GPU vs CPU performance

## 📊 Monitor GPU Usage

### Open Another Terminal and Run:
```cmd
nvidia-smi -l 1
```

### What to Look For:
```
+-----------------------------------------------------------------------------+
| GPU  Name            | GPU-Util | Memory-Usage |
|======================|==========|==============|
|   0  GTX 1050 Ti     |   75%    |  2500/4096MB |  ← Should increase!
+-----------------------------------------------------------------------------+
```

## 🎯 Expected Output

### With GPU Enabled (Option 1):
```
✓ XGBoost with GPU enabled
✓ LightGBM with GPU enabled

✓ Total models: 10
Models: ['XGBoost', 'LightGBM', 'LogisticRegression', ...]
```

### With GPU Enabled (Option 2):
```
✓ XGBoost available
✓ LightGBM available
ℹ CuML not available - using CPU fallback

Adding XGBoost GPU models...
Adding LightGBM GPU models...

✓ Total models to train: 12
Models: ['XGBoost_GPU', 'XGBoost_CPU', 'LightGBM_GPU', 'LightGBM_CPU', ...]

================================================================================
TRAINING MODELS (GPU-ACCELERATED)
================================================================================

Training: XGBoost_GPU
...
Time: 12.45s 🚀 (GPU)

Training: XGBoost_CPU
...
Time: 35.67s (CPU)

================================================================================
⚡ GPU SPEEDUP COMPARISON
================================================================================
XGBoost_GPU              GPU:  12.45s | CPU:  35.67s | Speedup: 2.86x
LightGBM_GPU             GPU:   8.23s | CPU:  24.11s | Speedup: 2.93x
```

## ⚙️ Configuration

### Fast Mode (Default):
```python
use_grid_search=False  # Quick training with default parameters
```

### Tuning Mode (Slower but Better):
```python
use_grid_search=True   # Full hyperparameter optimization
cv=5                   # 5-fold cross-validation
```

## 🔧 Troubleshooting

### "GPU not available" Message?
1. Make sure you have CUDA installed: `nvidia-smi`
2. Install GPU-enabled libraries:
   ```bash
   pip install xgboost lightgbm --upgrade
   ```

### GPU Utilization is 0%?
1. Check if models are actually GPU-enabled (look for "✓ GPU enabled" messages)
2. Dataset might be small (GPU overhead reduces benefit)
3. Try the full GPU cell (Option 2) which explicitly shows GPU/CPU comparison

### Error: "device='cuda' not supported"?
1. Your XGBoost might be CPU-only version
2. Reinstall: `pip uninstall xgboost && pip install xgboost --upgrade`

## 🎓 Which Option to Use?

### Use Option 1 (Simple Toggle) if:
- You want quick GPU training
- You don't need detailed GPU/CPU comparison
- You want minimal code changes

### Use Option 2 (Full GPU Cell) if:
- You want to see GPU vs CPU performance
- You want timing comparisons
- You want to maximize GPU utilization
- You need detailed performance metrics

## 🚀 Quick Start Commands

### 1. Verify GPU Setup:
```cmd
python D:\Work\Projects\LinkShield\check_gpu_setup.py
```

### 2. Monitor GPU (separate terminal):
```cmd
nvidia-smi -l 1
```

### 3. Run Notebook:
- Execute cells in order
- Choose Option 1 OR Option 2
- Watch GPU utilization increase!

## ✅ Summary

**Both cells are ready to use GPU!**

- **Option 1**: Simple `USE_GPU = True` toggle
- **Option 2**: Comprehensive GPU vs CPU comparison

Your GTX 1050 Ti will be utilized automatically when you run either cell! 🎉

