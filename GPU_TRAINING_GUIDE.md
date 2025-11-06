# GPU Training Guide for LinkShield

## Overview
The Manual Approach cell has been updated to support GPU-accelerated training for better performance. This guide explains how to enable and use GPU acceleration.

## GPU-Accelerated Libraries Supported

### 1. **XGBoost** (Recommended - Easiest to setup)
- **Installation**: `pip install xgboost`
- **GPU Usage**: Automatically uses GPU with `tree_method='gpu_hist'`
- **Requirements**: NVIDIA GPU with CUDA support
- **Performance**: 5-10x faster than CPU for large datasets

### 2. **LightGBM** (Good performance)
- **Installation**: `pip install lightgbm`
- **GPU Usage**: Set `device='gpu'`
- **Requirements**: OpenCL or CUDA support
- **Performance**: 3-8x faster than CPU

### 3. **CuML/RAPIDS** (Maximum GPU utilization)
- **Installation**: 
  - For CUDA 11: `pip install cuml-cu11 cupy-cuda11x`
  - For CUDA 12: `pip install cuml-cu12 cupy-cuda12x`
- **GPU Usage**: Native GPU implementations of sklearn algorithms
- **Requirements**: NVIDIA GPU with CUDA 11+ or 12+
- **Performance**: 10-50x faster for compatible algorithms

## Installation Steps

### Step 1: Install Basic GPU Libraries (Quick Start)
```bash
pip install xgboost lightgbm
```

This will enable XGBoost and LightGBM with GPU support. These are the easiest to set up.

### Step 2: Verify CUDA Installation (Optional but recommended)
Check if you have CUDA installed:
```bash
nvidia-smi
```

This should show your GPU information and CUDA version.

### Step 3: Install CuML for Maximum GPU Acceleration (Advanced)
Only if you need maximum performance:

For CUDA 11.x:
```bash
pip install cuml-cu11 cupy-cuda11x
```

For CUDA 12.x:
```bash
pip install cuml-cu12 cupy-cuda12x
```

## How the GPU Training Works

### Model Detection
The notebook automatically detects which GPU libraries are available:
- ✓ Shows green checkmark if library is available
- ✗ Shows red X with installation instructions if not available

### Model Selection
Based on available libraries, the code creates:

**If XGBoost is available:**
- `XGBoost_GPU` - GPU-accelerated XGBoost
- `XGBoost_CPU` - CPU version for comparison

**If LightGBM is available:**
- `LightGBM_GPU` - GPU-accelerated LightGBM
- `LightGBM_CPU` - CPU version for comparison

**If CuML is available:**
- `RandomForest_GPU` - GPU RandomForest
- `LogisticRegression_GPU` - GPU Logistic Regression
- `SVM_GPU` - GPU SVM
- `KNN_GPU` - GPU KNN

**Always included (CPU):**
- `DecisionTree` - CPU DecisionTree
- `GradientBoosting` - CPU GradientBoosting
- `AdaBoost` - CPU AdaBoost
- `GaussianNB` - CPU Naive Bayes

## Verifying GPU Usage

### Method 1: Check Training Output
The training function will show:
```
Training: XGBoost_GPU
...
Time: 15.23s (GPU)
```

Compare GPU vs CPU times to verify acceleration.

### Method 2: Monitor GPU Usage
In a separate terminal, run:
```bash
nvidia-smi -l 1
```

This refreshes GPU stats every second. You should see:
- **GPU-Util%** increase during training
- **Memory-Usage** increase when data is loaded to GPU

### Method 3: Compare Training Times
The results summary includes a "Time (s)" column showing training time for each model. GPU models should be significantly faster.

## Expected GPU Utilization

### XGBoost GPU
- **GPU Utilization**: 70-100% during tree building
- **Memory Usage**: Proportional to dataset size
- **Best For**: Tree-based models, large datasets

### LightGBM GPU
- **GPU Utilization**: 50-90% during training
- **Memory Usage**: Lower than XGBoost
- **Best For**: Fast training with reasonable memory

### CuML GPU
- **GPU Utilization**: 80-100% during computation
- **Memory Usage**: Higher than sklearn (needs cupy arrays)
- **Best For**: sklearn-compatible models, maximum acceleration

## Troubleshooting

### Issue: GPU Utilization is 0%

**Possible Causes:**
1. **Dataset too small**: GPU overhead exceeds benefits for very small datasets
   - **Solution**: Use CPU models for small datasets (<10K samples)

2. **GPU not properly configured**:
   - **Solution**: Reinstall xgboost/lightgbm with GPU support
   ```bash
   pip uninstall xgboost lightgbm
   pip install xgboost lightgbm --no-cache-dir
   ```

3. **CUDA version mismatch**:
   - **Solution**: Check CUDA version with `nvidia-smi` and install matching libraries

### Issue: "CUDA out of memory" Error

**Solutions:**
1. Reduce batch size in hyperparameter grid
2. Reduce `n_estimators` or `max_depth` parameters
3. Use CPU fallback models
4. Close other GPU-using applications

### Issue: CuML import fails

**Solutions:**
1. Verify CUDA installation: `nvidia-smi`
2. Install correct cuml version for your CUDA:
   ```bash
   # For CUDA 11
   pip install cuml-cu11 cupy-cuda11x
   
   # For CUDA 12
   pip install cuml-cu12 cupy-cuda12x
   ```
3. Use XGBoost/LightGBM instead if CuML installation fails

## Performance Tips

### 1. Start with Default Parameters
Run without grid search first to verify GPU is working:
```python
results, best_model, best_model_name = train_manual_classifier(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    model_type_list, parameter_list,
    use_grid_search=False  # Fast test
)
```

### 2. Use Grid Search for Final Model
Once verified, enable hyperparameter tuning:
```python
results, best_model, best_model_name = train_manual_classifier(
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    model_type_list, parameter_list,
    use_grid_search=True,  # Full tuning
    cv=5
)
```

### 3. Optimize Hyperparameter Grid
Reduce parameter combinations for faster results:
```python
parameter_list['XGBoost_GPU'] = {
    'n_estimators': [100, 200],  # Reduced from [100, 200, 300]
    'max_depth': [5, 7],         # Reduced from [3, 5, 7, 9]
    'learning_rate': [0.1, 0.3]  # Reduced from [0.01, 0.1, 0.3]
}
```

## Expected Performance Improvements

Based on your phishing dataset size:

| Model Type | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| XGBoost | ~60s | ~8-12s | 5-7x |
| LightGBM | ~45s | ~10-15s | 3-4x |
| RandomForest (CuML) | ~90s | ~5-10s | 9-18x |
| LogisticRegression (CuML) | ~15s | ~2-3s | 5-7x |

*Times are approximate and depend on dataset size and hardware*

## Recommendations

### For Quick Results (Recommended)
1. Install XGBoost and LightGBM only
2. Run with `use_grid_search=False` first
3. Monitor GPU usage with `nvidia-smi`

### For Maximum Performance
1. Install all libraries (XGBoost, LightGBM, CuML)
2. Start with default parameters
3. Use grid search on best performing models only

### For Production
1. Select the best performing GPU model
2. Fine-tune hyperparameters on that model only
3. Save the model with `joblib.dump()`

## Summary

The updated notebook now supports:
- ✅ Automatic GPU detection
- ✅ Multiple GPU-accelerated libraries
- ✅ CPU fallback when GPU unavailable
- ✅ Training time comparison (GPU vs CPU)
- ✅ Easy switching between GPU and CPU models

You should see GPU utilization when training models with `_GPU` suffix. If you don't see GPU usage, follow the troubleshooting steps above.

