"""
kernel_ref.py - Ground-truth reference unpacker and GEMV oracle for Samsung LittleBit.

Implements pure-PyTorch and register-level bitwise unpacking:
  - Bit extraction: (packed >> bit_idx) & 1
  - Value recovery: 1.0 - 2.0 * bit (bit 0 -> +1.0, bit 1 -> -1.0)
  - Column slice: ignores zero-padding beyond orig_cols
  - Vectorized GEMV and scalar register-unpacking simulation (CUDA warp/thread prototype)
  - Full LittleBitLinear forward pass reference
"""

from __future__ import annotations
from typing import Tuple, Optional, Any
import torch


def unpack_binary_matrix(
    packed_weight: torch.Tensor,
    orig_cols: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Unpacks a 2D uint32 or int32 packed matrix back to a {-1.0, +1.0} weight matrix.

    Bit order:
      Bit 0 (LSB) = column 0 within 32-element chunk
      Bit 31 (MSB) = column 31 within 32-element chunk
      bit 0 -> +1.0
      bit 1 -> -1.0

    Parameters
    ----------
    packed_weight : torch.Tensor
        2D tensor of shape (n_rows, words_per_row) with dtype uint32 or int32.
    orig_cols : int
        The unpadded number of columns in the original matrix.
    dtype : torch.dtype, default torch.float32
        Target floating point dtype (e.g. torch.float32 or torch.float16).

    Returns
    -------
    torch.Tensor
        Matrix of shape (n_rows, orig_cols) with values in {-1.0, +1.0}.
    """
    if packed_weight.ndim != 2:
        raise ValueError(f"Expected 2D packed tensor, got shape {packed_weight.shape}")

    n_rows, words_per_row = packed_weight.shape
    device = packed_weight.device

    # Cast to int64 for safe bit-shift operations without overflow or sign-extension issues
    # Masking with 0xFFFFFFFF ensures unsigned 32-bit interpretation
    packed_i64 = packed_weight.to(torch.int64) & 0xFFFFFFFF

    # 32 shifts: 0, 1, ..., 31
    shifts = torch.arange(32, dtype=torch.int64, device=device)

    # Broadcast: (n_rows, words_per_row, 1) >> (32,) -> (n_rows, words_per_row, 32)
    bits = (packed_i64.unsqueeze(-1) >> shifts) & 1

    # Flatten chunks along columns: (n_rows, words_per_row * 32)
    bits_flat = bits.view(n_rows, words_per_row * 32)

    # Slice out padding beyond orig_cols
    bits_valid = bits_flat[:, :orig_cols]

    # Convert bits to {-1.0, +1.0}: bit 0 -> +1.0, bit 1 -> -1.0
    weights = (1.0 - 2.0 * bits_valid.to(dtype))
    return weights.contiguous()


def unpack_and_matvec_ref(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    orig_cols: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Computes y = x @ W.T where W is stored in packed_weight (bit-packed {-1, +1}).
    
    Parameters
    ----------
    x : torch.Tensor
        Input activations of shape (..., orig_cols).
    packed_weight : torch.Tensor
        Packed weights of shape (out_features, words_per_row).
    orig_cols : int
        Number of unpadded columns (must match x.shape[-1]).
    dtype : torch.dtype, optional
        Computation/output dtype. Defaults to x.dtype.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (..., out_features).
    """
    target_dtype = dtype or x.dtype
    W = unpack_binary_matrix(packed_weight, orig_cols, dtype=target_dtype)
    return torch.matmul(x.to(target_dtype), W.t())


def matvec_register_sim(
    x_vec: torch.Tensor,
    packed_weight: torch.Tensor,
    orig_cols: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Exact simulation of a CUDA thread/warp register-unpacking inner loop.
    Demonstrates word-by-word register load and bitwise shift accumulation:
    
      acc = 0.0
      for word in row:
          reg = packed[row, word]
          for b in 0..31:
              col = word * 32 + b
              if col < orig_cols:
                  bit = (reg >> b) & 1
                  w = 1.0 - 2.0 * bit
                  acc += x[col] * w

    Parameters
    ----------
    x_vec : torch.Tensor
        1D or 2D input activations. If 2D (B, orig_cols), processes each row.
    packed_weight : torch.Tensor
        Packed weights (M, words_per_row).
    orig_cols : int
        Unpadded inner dimension.

    Returns
    -------
    torch.Tensor
        Output tensor (..., M).
    """
    orig_ndim = x_vec.ndim
    if orig_ndim == 1:
        x_2d = x_vec.unsqueeze(0)
    else:
        x_2d = x_vec.view(-1, orig_cols)

    batch_size = x_2d.shape[0]
    out_features, words_per_row = packed_weight.shape
    device = x_vec.device
    target_dtype = dtype or x_vec.dtype

    # Convert to CPU/NumPy or Python ints for bit-exact register simulation
    packed_cpu = packed_weight.cpu()
    x_cpu = x_2d.cpu().float()
    out = torch.zeros((batch_size, out_features), dtype=torch.float32)

    for b_idx in range(batch_size):
        x_row = x_cpu[b_idx]
        for m in range(out_features):
            acc = 0.0
            for w in range(words_per_row):
                # Load 32-bit word into simulated register
                val = int(packed_cpu[m, w].item()) & 0xFFFFFFFF
                for bit_idx in range(32):
                    col = w * 32 + bit_idx
                    if col >= orig_cols:
                        break
                    bit = (val >> bit_idx) & 1
                    w_val = 1.0 - 2.0 * bit
                    acc += float(x_row[col]) * w_val
            out[b_idx, m] = acc

    out = out.to(device=device, dtype=target_dtype)
    if orig_ndim == 1:
        return out.squeeze(0)
    return out.view(*x_vec.shape[:-1], out_features)


def littlebit_forward_ref(
    x: torch.Tensor,
    layer: Any,
    use_register_sim: bool = False,
) -> torch.Tensor:
    """
    Full reference forward pass for a bit-packed LittleBit layer.
    
    Forward Equation:
      y_main = ((((x * v2) @ V.T) * (v1 * u2)) @ U.T) * u1
      if residual:
        y_res = ((((x * v2_R) @ V_R.T) * (v1_R * u2_R)) @ U_R.T) * u1_R
        y = y_main + y_res
      if bias:
        y += bias

    Parameters
    ----------
    x : torch.Tensor
        Input activations of shape (..., in_features).
    layer : PackedLittleBitLayer or dict
        Packed layer containing packed weights and scales.
    use_register_sim : bool, default False
        If True, uses the scalar register-simulation loop (for verification).
        If False, uses the fast vectorized bitwise reference.

    Returns
    -------
    torch.Tensor
        Layer output of shape (..., out_features) matching x.dtype.
    """
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    in_features = getattr(layer, "in_features", layer["in_features"])
    out_features = getattr(layer, "out_features", layer["out_features"])
    split_dim = getattr(layer, "split_dim", layer["split_dim"])
    residual = getattr(layer, "residual", layer["residual"])

    if hidden_dim != in_features:
        raise ValueError(f"Input dim {hidden_dim} does not match layer in_features {in_features}")

    x_flat = x.view(-1, hidden_dim)
    orig_dtype = x.dtype
    calc_dtype = torch.float32  # Reference calculates in float32 for maximum precision

    gemv_fn = matvec_register_sim if use_register_sim else unpack_and_matvec_ref

    # 1. Main Branch Forward
    v2 = layer["v2"].to(device=x.device, dtype=calc_dtype)
    v1 = layer["v1"].to(device=x.device, dtype=calc_dtype)
    u2 = layer["u2"].to(device=x.device, dtype=calc_dtype)
    u1 = layer["u1"].to(device=x.device, dtype=calc_dtype)
    V_packed = layer["V_packed"].to(device=x.device)
    U_packed = layer["U_packed"].to(device=x.device)

    # Step 1: scale input
    a = x_flat.to(calc_dtype) * v2
    # Step 2: first binary matmul with V.T (in_features -> split_dim)
    h = gemv_fn(a, V_packed, orig_cols=in_features, dtype=calc_dtype)
    # Step 3: scale latent representation
    g = h * (v1 * u2)
    # Step 4: second binary matmul with U.T (split_dim -> out_features)
    z = gemv_fn(g, U_packed, orig_cols=split_dim, dtype=calc_dtype)
    # Step 5: scale output
    y = z * u1

    # 2. Residual Branch (if present)
    if residual and ("V_R_packed" in layer or hasattr(layer, "V_R_packed")):
        v2_R = layer["v2_R"].to(device=x.device, dtype=calc_dtype)
        v1_R = layer["v1_R"].to(device=x.device, dtype=calc_dtype)
        u2_R = layer["u2_R"].to(device=x.device, dtype=calc_dtype)
        u1_R = layer["u1_R"].to(device=x.device, dtype=calc_dtype)
        V_R_packed = layer["V_R_packed"].to(device=x.device)
        U_R_packed = layer["U_R_packed"].to(device=x.device)

        a_R = x_flat.to(calc_dtype) * v2_R
        h_R = gemv_fn(a_R, V_R_packed, orig_cols=in_features, dtype=calc_dtype)
        g_R = h_R * (v1_R * u2_R)
        z_R = gemv_fn(g_R, U_R_packed, orig_cols=split_dim, dtype=calc_dtype)
        y_R = z_R * u1_R
        y = y + y_R

    # 3. Add Bias (if present)
    if ("bias" in layer or hasattr(layer, "bias")) and layer["bias"] is not None:
        bias = layer["bias"].to(device=x.device, dtype=calc_dtype)
        y = y + bias

    # Reshape back to prefix batch dimensions
    output_shape = (*orig_shape[:-1], out_features)
    return y.to(orig_dtype).view(output_shape)
