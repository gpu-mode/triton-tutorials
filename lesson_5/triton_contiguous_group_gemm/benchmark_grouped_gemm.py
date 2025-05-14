# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
import triton
import triton.language as tl
from typing import Tuple
from grouped_gemm import grouped_gemm_persistent


def construct_grouped_gemm(
    M: int,
    K: int,
    N: int,
    num_experts: int,
    group_size_m: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test data with proper block alignment.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension (K)
        output_dim: Output dimension (N)
        num_experts: Number of experts
        group_size_m: Size of expert groups
        device: Device to create tensors on
        dtype: Data type for inputs and weights

    Returns:
        Tuple of (inputs, expert_weights, expert_indices)
    """
    # Calculate total number of tokens
    M_total = M * num_experts

    # Ensure M_total is a multiple of group_size_m
    padded_M = ((M_total + group_size_m - 1) // group_size_m) * group_size_m
    padding_needed = padded_M - M_total

    if padding_needed > 0:
        print(f"Padding input from {M_total} to {padded_M} to ensure group alignment")
        M_total = padded_M

    # Create inputs
    inputs = torch.randn((M_total, K), dtype=dtype, device=device)

    # Create expert weights
    expert_weights = torch.randn(
        (num_experts, N, K), dtype=dtype, device=device
    )

    # Create expert indices with proper group alignment
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)

    # Assign experts in contiguous blocks of group_size_m
    num_groups = M_total // group_size_m

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size_m
        end_idx = start_idx + group_size_m

        # Assign this entire group to one expert
        expert_idx = group_idx % num_experts
        expert_indices[start_idx:end_idx] = expert_idx

    return inputs, expert_weights, expert_indices


def pytorch_reference_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Reference implementation using PyTorch for verification.
    """
    M_total, K = inputs.shape
    num_experts, N, _ = expert_weights.shape

    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Process each group
    for i in range(0, M_total, group_size_m):
        end_idx = min(i + group_size_m, M_total)

        # Get expert index for this group
        expert_idx = expert_indices[i].item()

        # Get expert weights
        expert_weight = expert_weights[expert_idx]

        # Compute output for this group
        output[i:end_idx] = torch.matmul(inputs[i:end_idx], expert_weight.t())

    return output

# PyTorch Benchmarking
import torch.utils.benchmark as benchmark


def triton_gemm_func(a, b, expert_indices):
    return grouped_gemm_persistent(a, b, expert_indices)

def gemm_func_torch(a, b, expert_indices):
    return pytorch_reference_gemm(a, b, expert_indices)

num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')
results = []

for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):

    a, b, expert_indices = construct_grouped_gemm(m, k, n, num_groups)

    label = 'BF16 Grouped GEMM Performance'
    sub_label = f'num_groups: {num_groups}, m: {m}, n: {n}, k: {k}'

    results.append(benchmark.Timer(
        stmt='triton_gemm_func(a, b, expert_indices)',
        setup='from __main__ import triton_gemm_func',
        globals={'a': a, 'b' : b, 'expert_indices': expert_indices},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='Triton Group GEMM').blocked_autorange(min_run_time=1))

    results.append(benchmark.Timer(
        stmt='gemm_func_torch(a, b, m_offsets)',
        setup='from __main__ import gemm_func_torch',
        globals={'a': a, 'b' : b, 'expert_indices': expert_indices},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description='PyTorch Reference Group GEMM').blocked_autorange(min_run_time=1))
    
    
compare = benchmark.Compare(results)
compare.print()