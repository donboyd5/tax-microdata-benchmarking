# Add GPU Acceleration for National Weight Optimization

## Summary

This PR adds optional GPU acceleration to the national weight optimization process in `tmd/utils/reweight.py`, providing **5.7x performance improvement** on CUDA-capable systems while maintaining full backward compatibility.

## Performance Results

**Benchmark Results** (NVIDIA GeForce GTX 1070 Ti):
- **CPU (Master)**: 4m20s optimization time, ~7.7 iterations/second
- **GPU (This PR)**: 46s optimization time, ~50 iterations/second  
- **Overall Speedup**: **5.7x faster**

## Features Added

### 🚀 Automatic GPU Detection
- Detects CUDA availability automatically
- Falls back to CPU gracefully if GPU unavailable
- No configuration required

### ⚙️ User Control
- New optional parameter `use_gpu: bool = True` in `reweight()` function
- Users can disable GPU even if available: `reweight(data, use_gpu=False)`
- Informative logging shows which device is being used

### 🛡️ Robust Error Handling
All scenarios handled gracefully:
- ✅ GPU available + enabled → Uses GPU
- ✅ GPU available + disabled by user → Uses CPU  
- ✅ GPU unavailable + requested → Uses CPU (with warning)
- ✅ GPU unavailable → Uses CPU

### 🔍 Clear User Feedback
```
...GPU acceleration enabled: NVIDIA GeForce GTX 1070 Ti (8.0 GB)
...GPU available but disabled by user, using CPU  
...GPU requested but not available, using CPU
...GPU not available, using CPU
```

## Technical Implementation

### Changes Made
1. **Enhanced `reweight()` function signature**:
   ```python
   def reweight(
       flat_file: pd.DataFrame,
       time_period: int = 2021,
       weight_multiplier_min: float = REWEIGHT_MULTIPLIER_MIN,
       weight_multiplier_max: float = REWEIGHT_MULTIPLIER_MAX, 
       weight_deviation_penalty: float = REWEIGHT_DEVIATION_PENALTY,
       use_gpu: bool = True,  # NEW PARAMETER
   ):
   ```

2. **Intelligent device selection**:
   ```python
   gpu_available = torch.cuda.is_available()
   use_gpu_actual = use_gpu and gpu_available
   device = torch.device("cuda" if use_gpu_actual else "cpu")
   ```

3. **Direct tensor creation on device**:
   ```python
   # Avoids "non-leaf tensor" optimization errors
   weights = torch.tensor(data, dtype=torch.float32, device=device)
   weight_multiplier = torch.tensor(data, device=device, requires_grad=True)
   ```

4. **Proper CPU conversion for numpy**:
   ```python
   final_weights = new_weights.detach().cpu().numpy()
   ```

### Compatibility
- **Zero breaking changes** - all existing code works unchanged
- **Backward compatible** - `use_gpu` parameter is optional with sensible default
- **Cross-platform** - works on systems with or without CUDA support
- **Dependencies** - no new dependencies required (PyTorch already supports CUDA)

## Testing

### Test Scenarios Verified
- [x] GPU acceleration working (5.7x speedup achieved)
- [x] CPU fallback when `use_gpu=False` 
- [x] CPU fallback when CUDA unavailable
- [x] Identical numerical results between CPU and GPU
- [x] Error handling for all device scenarios
- [x] Proper memory management (no GPU memory leaks)

### Benchmark Logs
- `gpu_test_v2.log` - Successful GPU acceleration test
- `gpu_disabled_test.log` - CPU fallback verification
- `data_master_test2.log` - CPU baseline for comparison

## Impact

### Performance Benefits
- **Dramatic speedup** for users with CUDA-capable GPUs
- **No performance penalty** for users without GPUs
- **Scales with GPU capability** - better GPUs = better performance

### User Experience
- **Transparent acceleration** - works automatically
- **User control** - can be disabled if needed
- **Clear feedback** - users know which device is being used
- **Zero configuration** - no setup required

## Implementation Notes

This implementation follows PyTorch best practices for device-agnostic code and includes comprehensive error handling. The performance improvement comes from leveraging GPU parallelization for the 2000-iteration Adam optimization loop that dominates the reweighting computation.

The code maintains identical numerical accuracy between CPU and GPU modes while providing substantial performance gains for users with appropriate hardware.

## Files Changed
- `tmd/utils/reweight.py` - Main GPU acceleration implementation
- `tmd/datasets/tmd.py` - Function call (no breaking changes)

## Backward Compatibility
✅ All existing code continues to work unchanged  
✅ Optional parameter with sensible default  
✅ Graceful fallback behavior  
✅ No new dependencies required