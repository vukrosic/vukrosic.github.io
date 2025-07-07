---
published: true
---

# DeepGEMM Essentials MD: High-Performance FP8 Matrix Multiplication

## Part 1: Getting Started - Your First FP8 GEMM

```python
import torch
import deep_gemm

# Create simple input matrices
m, n, k = 128, 256, 512
lhs = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
rhs = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
output = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

print(f"LHS shape: {lhs.shape}")    # [128, 512]
print(f"RHS shape: {rhs.shape}")    # [256, 512] 
print(f"Output shape: {output.shape}")  # [128, 256]
```

**What happened**: We created the basic tensors for matrix multiplication: LHS × RHS^T = Output.

## Part 2: Understanding FP8 - Why It Matters

```python
# Regular BF16 GEMM (what you normally do)
reference = lhs @ rhs.t()  # Standard PyTorch GEMM

# Check memory usage
bf16_memory = lhs.numel() * 2 + rhs.numel() * 2  # 2 bytes per BF16
fp8_memory = lhs.numel() * 1 + rhs.numel() * 1   # 1 byte per FP8

print(f"BF16 memory: {bf16_memory / 1024**2:.1f} MB")
print(f"FP8 memory: {fp8_memory / 1024**2:.1f} MB")
print(f"Memory saved: {(1 - fp8_memory/bf16_memory)*100:.1f}%")
```

**Key insight**: FP8 uses half the memory while maintaining good accuracy with proper scaling.

## Part 3: Converting to FP8 with Scaling

```python
def cast_to_fp8_per_token(x: torch.Tensor):
    """Convert tensor to FP8 with per-token (per-row) scaling"""
    assert x.dim() == 2
    m, n = x.shape
    
    # Pad to 128-element boundaries (FP8 requirement)
    pad_size = (128 - (n % 128)) % 128
    if pad_size > 0:
        x = torch.nn.functional.pad(x, (0, pad_size), value=0)
    
    # Reshape for scaling calculation
    x_view = x.view(m, -1, 128)  # [m, n/128, 128]
    
    # Find max absolute value per 128-element block
    x_amax = x_view.abs().float().amax(dim=2).clamp(1e-4)  # [m, n/128]
    
    # Scale to FP8 range (448.0 is max representable value)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    scale_factors = (x_amax / 448.0)
    
    return fp8_data.view(m, -1)[:, :n], scale_factors

# Convert our matrices
lhs_fp8, lhs_scales = cast_to_fp8_per_token(lhs)
print(f"Original: {lhs.dtype}, Converted: {lhs_fp8.dtype}")
print(f"Scale factors shape: {lhs_scales.shape}")
```

**Critical concept**: Scaling prevents overflow and maintains precision in the limited FP8 range.

## Part 4: Block-wise Scaling for RHS

```python
def cast_to_fp8_per_block(x: torch.Tensor):
    """Convert tensor to FP8 with per-block scaling (128x128 blocks)"""
    m, n = x.shape
    
    # Pad to 128x128 blocks
    padded_m = ((m + 127) // 128) * 128
    padded_n = ((n + 127) // 128) * 128
    
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    
    # Reshape into 128x128 blocks
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    
    # Find max per block
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    
    # Scale to FP8
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    scale_factors = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    
    return x_scaled.view_as(x_padded)[:m, :n], scale_factors

# Convert RHS with block scaling
rhs_fp8, rhs_scales = cast_to_fp8_per_block(rhs)
print(f"RHS FP8 shape: {rhs_fp8.shape}")
print(f"RHS scales shape: {rhs_scales.shape}")
```

**Why different scaling**: LHS uses fine-grained scaling, RHS uses coarser blocks for efficiency.

## Part 5: Preparing Tensors for DeepGEMM

```python
# DeepGEMM requires specific tensor layouts
from deep_gemm import get_col_major_tma_aligned_tensor

# LHS scales must be transposed and TMA-aligned
lhs_scales_aligned = get_col_major_tma_aligned_tensor(lhs_scales)

# RHS scales must be contiguous
assert rhs_scales.is_contiguous()

# Package the inputs
lhs_input = (lhs_fp8, lhs_scales_aligned)
rhs_input = (rhs_fp8, rhs_scales)

print("✓ Tensors prepared for DeepGEMM")
print(f"LHS scales alignment: {lhs_scales_aligned.stride()}")
```

**TMA requirement**: Tensor Memory Accelerator needs specific memory alignment for optimal performance.

## Part 6: Your First DeepGEMM Call

```python
# Perform the FP8 GEMM
deep_gemm.gemm_fp8_fp8_bf16_nt(lhs_input, rhs_input, output)

# Verify correctness
reference = lhs @ rhs.t()
error = torch.abs(output - reference).max().item()
relative_error = (error / torch.abs(reference).max().item()) * 100

print(f"Max absolute error: {error:.6f}")
print(f"Relative error: {relative_error:.3f}%")
print("✓ FP8 GEMM completed successfully!")
```

**Result**: High-performance FP8 matrix multiplication with automatic kernel optimization.

## Part 7: Understanding the Performance Gain

```python
import time

def benchmark_gemm(func, *args, num_runs=10):
    # Warmup
    for _ in range(3):
        func(*args)
    torch.cuda.synchronize()
    
    # Timing
    start = time.time()
    for _ in range(num_runs):
        func(*args)
    torch.cuda.synchronize()
    
    return (time.time() - start) / num_runs

# Benchmark both versions
fp8_time = benchmark_gemm(deep_gemm.gemm_fp8_fp8_bf16_nt, lhs_input, rhs_input, output)
bf16_time = benchmark_gemm(lambda x, y, out: out.copy_(x @ y.t()), lhs, rhs, reference)

# Calculate throughput (TFLOPS)
ops = 2 * m * n * k  # Multiply-accumulate operations
fp8_tflops = ops / fp8_time / 1e12
bf16_tflops = ops / bf16_time / 1e12

print(f"FP8 GEMM:  {fp8_time*1000:.2f}ms ({fp8_tflops:.1f} TFLOPS)")
print(f"BF16 GEMM: {bf16_time*1000:.2f}ms ({bf16_tflops:.1f} TFLOPS)")
print(f"Speedup: {bf16_time/fp8_time:.1f}x")
```

**Performance**: FP8 can achieve 2-3x speedup on modern GPUs while using half the memory.

## Part 8: Grouped GEMM - Processing Multiple Experts

```python
# Simulate MoE (Mixture of Experts) scenario
num_experts = 4
tokens_per_expert = [128, 96, 112, 144]  # Variable tokens per expert
expert_dim = 512

# Create contiguous tensor for all tokens
total_tokens = sum(tokens_per_expert)
alignment = deep_gemm.get_m_alignment_for_contiguous_layout()  # 128

# Align each expert's token count
aligned_tokens = [((t + alignment - 1) // alignment) * alignment for t in tokens_per_expert]
total_aligned = sum(aligned_tokens)

print(f"Original tokens: {tokens_per_expert}")
print(f"Aligned tokens: {aligned_tokens}")
print(f"Total aligned: {total_aligned}")
```

**MoE insight**: Different experts process different numbers of tokens - grouping improves efficiency.

## Part 9: Setting Up Grouped GEMM Data

```python
# Create inputs for grouped GEMM
lhs_grouped = torch.randn((total_aligned, k), device='cuda', dtype=torch.bfloat16)
rhs_grouped = torch.randn((num_experts, n, k), device='cuda', dtype=torch.bfloat16)
output_grouped = torch.empty((total_aligned, n), device='cuda', dtype=torch.bfloat16)

# Create mapping tensor
m_indices = torch.empty(total_aligned, device='cuda', dtype=torch.int32)
start = 0
for expert_id, (orig_tokens, aligned_tokens) in enumerate(zip(tokens_per_expert, aligned_tokens)):
    # Real tokens get expert ID
    m_indices[start:start + orig_tokens] = expert_id
    # Padding tokens get -1 (ignored)
    m_indices[start + orig_tokens:start + aligned_tokens] = -1
    start += aligned_tokens

print(f"Mapping tensor shape: {m_indices.shape}")
print(f"Expert assignments: {m_indices[:20]}")  # First 20 tokens
```

**Mapping**: Each token knows which expert should process it.

## Part 10: Converting Grouped Data to FP8

```python
# Convert LHS (same as before)
lhs_grouped_fp8, lhs_grouped_scales = cast_to_fp8_per_token(lhs_grouped)
lhs_grouped_scales = get_col_major_tma_aligned_tensor(lhs_grouped_scales)

# Convert each expert's RHS separately
rhs_grouped_fp8 = torch.empty_like(rhs_grouped, dtype=torch.float8_e4m3fn)
rhs_grouped_scales = torch.empty((num_experts, (n + 127) // 128, (k + 127) // 128), 
                                device='cuda', dtype=torch.float32)

for expert_id in range(num_experts):
    rhs_grouped_fp8[expert_id], rhs_grouped_scales[expert_id] = cast_to_fp8_per_block(rhs_grouped[expert_id])

# Package inputs
lhs_grouped_input = (lhs_grouped_fp8, lhs_grouped_scales)
rhs_grouped_input = (rhs_grouped_fp8, rhs_grouped_scales)

print("✓ Grouped data converted to FP8")
```

**Expert-wise**: Each expert has its own scaling factors for optimal precision.

## Part 11: Running Grouped GEMM

```python
# Perform grouped GEMM
deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
    lhs_grouped_input, 
    rhs_grouped_input, 
    output_grouped, 
    m_indices
)

# Verify by computing reference
reference_grouped = torch.zeros_like(output_grouped)
start = 0
for expert_id, aligned_tokens in enumerate(aligned_tokens):
    end = start + aligned_tokens
    reference_grouped[start:end] = lhs_grouped[start:end] @ rhs_grouped[expert_id].t()
    start = end

# Mask out padding tokens for comparison
valid_mask = (m_indices != -1).unsqueeze(1)
output_masked = torch.where(valid_mask, output_grouped, torch.zeros_like(output_grouped))
reference_masked = torch.where(valid_mask, reference_grouped, torch.zeros_like(reference_grouped))

error = torch.abs(output_masked - reference_masked).max().item()
print(f"Grouped GEMM error: {error:.6f}")
print("✓ Grouped GEMM completed successfully!")
```

**Validation**: Compare against standard computation to ensure correctness.

## Part 12: Weight Gradient GEMM

```python
# For training: compute weight gradients
def setup_weight_gradient():
    m_grad, k_grad, n_grad = 256, 1024, 512
    
    # Activations (forward pass)
    activations = torch.randn((m_grad, k_grad), device='cuda', dtype=torch.bfloat16)
    
    # Gradient w.r.t. output (from backprop)
    grad_output = torch.randn((m_grad, n_grad), device='cuda', dtype=torch.bfloat16)
    
    # Weight gradient accumulator (typically has residual)
    weight_grad = torch.randn((n_grad, k_grad), device='cuda', dtype=torch.float) * 0.1
    
    return activations, grad_output, weight_grad

activations, grad_output, weight_grad = setup_weight_gradient()

# Convert to FP8
act_fp8, act_scales = cast_to_fp8_per_token(activations)
grad_fp8, grad_scales = cast_to_fp8_per_token(grad_output)

# Prepare inputs (both need transposed scales)
act_input = (act_fp8, get_col_major_tma_aligned_tensor(act_scales))
grad_input = (grad_fp8, get_col_major_tma_aligned_tensor(grad_scales))

print(f"Weight gradient shape: {weight_grad.shape}")
print(f"Accumulator dtype: {weight_grad.dtype}")  # FP32 for precision
```

**Training context**: Weight gradients accumulate many small updates - need FP32 precision.

## Part 13: Computing Weight Gradients

```python
# Compute weight gradients with accumulation
original_grad = weight_grad.clone()

deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(grad_input, act_input, weight_grad)

# Verify: grad_output^T @ activations + original_grad
reference_update = grad_output.float().t() @ activations.float()
expected_grad = original_grad + reference_update

error = torch.abs(weight_grad - expected_grad).max().item()
relative_error = error / torch.abs(expected_grad).max().item()

print(f"Weight gradient error: {error:.6f}")
print(f"Relative error: {relative_error*100:.3f}%")
print("✓ Weight gradient computation successful!")
```

**Accumulation**: New gradients are added to existing values, enabling mini-batch training.

## Part 14: Performance Monitoring

```python
def analyze_gemm_performance(m, n, k, operation="forward"):
    # Theoretical peak performance
    # H100 has ~1600 TFLOPS FP8 peak
    ops = 2 * m * n * k
    peak_time = ops / 1600e12  # Theoretical minimum time
    
    # Memory bandwidth
    fp8_bytes = (m * k + n * k) * 1  # FP8 inputs
    bf16_bytes = m * n * 2           # BF16 output  
    scale_bytes = ((m * k) // 128 + (n * k) // 128) * 4  # FP32 scales
    total_bytes = fp8_bytes + bf16_bytes + scale_bytes
    
    # H100 has ~3TB/s memory bandwidth
    bandwidth_time = total_bytes / 3e12
    
    print(f"\n{operation.upper()} GEMM Analysis (M={m}, N={n}, K={k}):")
    print(f"Operations: {ops/1e9:.1f} GigaOps")
    print(f"Compute bound time: {peak_time*1000:.2f}ms")
    print(f"Memory bound time: {bandwidth_time*1000:.2f}ms")
    print(f"Bottleneck: {'Compute' if peak_time > bandwidth_time else 'Memory'}")

# Analyze our configurations
analyze_gemm_performance(128, 256, 512, "forward")
analyze_gemm_performance(256, 512, 1024, "weight_grad")
```

**Performance tuning**: Understanding compute vs memory bottlenecks helps optimize configurations.

## Part 15: Advanced Configuration

```python
# Control SM utilization for better efficiency
original_sms = deep_gemm.get_num_sms()
print(f"Default SMs: {original_sms}")

# Use fewer SMs for smaller problems to save power
deep_gemm.set_num_sms(original_sms // 2)
print(f"Reduced SMs: {deep_gemm.get_num_sms()}")

# Run a smaller GEMM
small_lhs = torch.randn((64, 256), device='cuda', dtype=torch.bfloat16)
small_rhs = torch.randn((128, 256), device='cuda', dtype=torch.bfloat16)
small_out = torch.empty((64, 128), device='cuda', dtype=torch.bfloat16)

small_lhs_fp8, small_lhs_scales = cast_to_fp8_per_token(small_lhs)
small_rhs_fp8, small_rhs_scales = cast_to_fp8_per_block(small_rhs)

deep_gemm.gemm_fp8_fp8_bf16_nt(
    (small_lhs_fp8, get_col_major_tma_aligned_tensor(small_lhs_scales)),
    (small_rhs_fp8, small_rhs_scales),
    small_out
)

# Restore original setting
deep_gemm.set_num_sms(original_sms)
print("✓ SM configuration demonstrated")
```

**Resource management**: Control GPU utilization for power efficiency and multi-tenancy.

## Part 16: Debugging and Validation

```python
def validate_fp8_conversion(original, fp8_data, scales):
    """Check if FP8 conversion preserves data accurately"""
    # Reconstruct original from FP8
    if fp8_data.dim() == 2:
        # Per-token scaling
        m, n = fp8_data.shape
        fp8_view = fp8_data.view(m, -1, 128)
        scales_expanded = scales.unsqueeze(2)
        reconstructed = fp8_view.float() * scales_expanded
        reconstructed = reconstructed.view(m, -1)[:, :original.shape[1]]
    
    # Compare
    abs_error = torch.abs(original.float() - reconstructed).max().item()
    rel_error = abs_error / torch.abs(original.float()).max().item()
    
    print(f"FP8 conversion error: {abs_error:.6f} ({rel_error*100:.3f}%)")
    return abs_error < 1e-2  # Reasonable threshold for FP8

# Validate our conversions
lhs_valid = validate_fp8_conversion(lhs, lhs_fp8, lhs_scales)
rhs_valid = validate_fp8_conversion(rhs, rhs_fp8[0], rhs_scales[0])

print(f"LHS conversion valid: {lhs_valid}")
print(f"RHS conversion valid: {rhs_valid}")
```

**Quality assurance**: Always validate FP8 conversions to ensure acceptable precision loss.

## Part 17: Memory Optimization

```python
def estimate_memory_usage(shapes, operation="gemm"):
    """Estimate GPU memory usage for DeepGEMM operations"""
    m, n, k = shapes
    
    # Input tensors
    lhs_fp8 = m * k * 1          # FP8
    lhs_scales = m * ((k + 127) // 128) * 4  # FP32
    rhs_fp8 = n * k * 1          # FP8  
    rhs_scales = ((n + 127) // 128) * ((k + 127) // 128) * 4  # FP32
    
    # Output
    if operation == "gemm":
        output = m * n * 2       # BF16
    else:  # weight_grad
        output = m * n * 4       # FP32
    
    # Temporary workspace (estimated)
    workspace = max(m, n) * 1024 * 4  # Conservative estimate
    
    total = lhs_fp8 + lhs_scales + rhs_fp8 + rhs_scales + output + workspace
    
    print(f"Memory usage for {shapes}:")
    print(f"  Inputs: {(lhs_fp8 + lhs_scales + rhs_fp8 + rhs_scales) / 1024**2:.1f} MB")
    print(f"  Output: {output / 1024**2:.1f} MB")
    print(f"  Workspace: {workspace / 1024**2:.1f} MB")
    print(f"  Total: {total / 1024**2:.1f} MB")
    
    return total

# Estimate for different problem sizes
estimate_memory_usage((1024, 2048, 4096), "gemm")
estimate_memory_usage((2048, 4096, 8192), "weight_grad")
```

**Capacity planning**: Understand memory requirements for different model sizes.

## Part 18: Integration with Training Loops

```python
class FP8LinearLayer:
    """Example of integrating DeepGEMM into a training loop"""
    
    def __init__(self, in_features, out_features):
        self.weight = torch.randn((out_features, in_features), 
                                device='cuda', dtype=torch.bfloat16)
        self.weight_grad = torch.zeros_like(self.weight, dtype=torch.float)
    
    def forward(self, x):
        # Convert inputs to FP8
        x_fp8, x_scales = cast_to_fp8_per_token(x)
        w_fp8, w_scales = cast_to_fp8_per_block(self.weight)
        
        # Prepare DeepGEMM inputs
        x_input = (x_fp8, get_col_major_tma_aligned_tensor(x_scales))
        w_input = (w_fp8, w_scales)
        
        # Allocate output
        output = torch.empty((x.shape[0], self.weight.shape[0]), 
                           device='cuda', dtype=torch.bfloat16)
        
        # Forward pass
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_input, w_input, output)
        return output
    
    def backward(self, x, grad_output):
        # Convert to FP8
        x_fp8, x_scales = cast_to_fp8_per_token(x)
        grad_fp8, grad_scales = cast_to_fp8_per_token(grad_output)
        
        # Prepare inputs
        x_input = (x_fp8, get_col_major_tma_aligned_tensor(x_scales))
        grad_input = (grad_fp8, get_col_major_tma_aligned_tensor(grad_scales))
        
        # Compute weight gradients: grad_output^T @ x
        deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(grad_input, x_input, self.weight_grad)

# Demo usage
layer = FP8LinearLayer(512, 256)
x = torch.randn((128, 512), device='cuda', dtype=torch.bfloat16)

# Forward pass
y = layer.forward(x)
print(f"Forward output shape: {y.shape}")

# Backward pass
grad_y = torch.randn_like(y)
layer.backward(x, grad_y)
print(f"Weight grad shape: {layer.weight_grad.shape}")
print("✓ Training loop integration demonstrated")
```

**Real-world usage**: How to integrate DeepGEMM into actual neural network training.

## Key Takeaways

1. **FP8 = 2x Memory Savings**: Half the storage with proper scaling
2. **Scaling is Critical**: Per-token and per-block strategies maintain precision
3. **TMA Alignment**: Required for optimal hardware utilization
4. **Grouped Operations**: Efficient for MoE and variable-size batches
5. **JIT Compilation**: Automatic kernel optimization for each shape
6. **Memory Layout Matters**: Column-major scales, contiguous tensors
7. **FP32 Accumulation**: Use higher precision for gradients

## Practice Challenge

```python
# Create an MoE layer with 8 experts
# Process a batch with variable expert utilization
# Measure memory savings vs standard implementation

num_experts = 8
expert_tokens = [64, 128, 96, 112, 88, 144, 72, 104]  # Realistic distribution
hidden_dim = 2048

# Your implementation here:
# 1. Set up grouped GEMM inputs
# 2. Convert to FP8 
# 3. Run DeepGEMM
# 4. Compare with standard PyTorch
# 5. Measure performance and memory usage

print("Challenge: Implement efficient MoE with DeepGEMM!")
```

Master these concepts and you'll be able to leverage cutting-edge FP8 acceleration for transformer training and inference at scale!