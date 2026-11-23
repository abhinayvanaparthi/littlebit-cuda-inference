"""
packer.py - Bit-packing runtime utilities for Samsung LittleBit models.

Converts {-1.0, +1.0} 2D tensors into tightly bit-packed uint32 tensors:
  - 32 weights per uint32 word
  - bit 0 = LSB (corresponds to col 0 in each 32-element chunk)
  - +1.0 -> bit 0
  - -1.0 -> bit 1
  - Inner dimensions not divisible by 32 are zero-padded in bit domain
    (zero bit = +1.0 weight) to the next multiple of 32.
  - orig_in_features is recorded in metadata and on the tensor.

Provides:
  - pack_binary_matrix(weight_pm1, return_orig_dim=False)
  - export_layer_to_safetensors(layer, file_path)
  - load_packed_layer(file_path, device="cpu")
  - PackedLittleBitLayer (PyTorch nn.Module container with dict access)
"""

from __future__ import annotations
import os
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from safetensors import safe_open


def pack_binary_matrix(
    weight_pm1: torch.Tensor,
    return_orig_dim: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Packs a 2D {-1.0, +1.0} weight tensor into a 2D uint32 tensor.
    
    Encoding:
      +1.0 -> bit 0
      -1.0 -> bit 1
      Bit 0 (LSB) = column 0 of each 32-column slice.
      Bit 31 (MSB) = column 31 of each 32-column slice.
      
    Padding:
      If n_cols is not a multiple of 32, zero-pads the bit domain to the
      next multiple of 32 (padding bits are 0, which corresponds to +1.0).
      
    Parameters
    ----------
    weight_pm1 : torch.Tensor
        2D tensor of shape (n_rows, n_cols) with values in {-1.0, +1.0}.
        If unbinarized float values are passed, sign() is used.
    return_orig_dim : bool, default False
        If True, returns (packed_tensor, orig_in_features).
        If False, returns packed_tensor with orig_in_features set as an attribute.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, int]
        Packed tensor of shape (n_rows, ceil(n_cols / 32)) with dtype torch.uint32.
    """
    if weight_pm1.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {weight_pm1.shape}")

    n_rows, n_cols = weight_pm1.shape
    device = weight_pm1.device

    # Binarize if not strictly {-1, +1}: map negative to 1, non-negative to 0
    # +1.0 -> 0, -1.0 -> 1
    bits = (weight_pm1 < 0).to(torch.int64)

    words_per_row = (n_cols + 31) // 32
    padded_cols = words_per_row * 32
    pad = padded_cols - n_cols

    if pad > 0:
        # Zero-pad bits: bit 0 corresponds to +1.0
        pad_bits = torch.zeros((n_rows, pad), dtype=torch.int64, device=device)
        bits = torch.cat([bits, pad_bits], dim=1)

    # Group into chunks of 32 bits along the column dimension
    bits_grouped = bits.view(n_rows, words_per_row, 32)

    # Powers of 2 for LSB-first ordering (col 0 -> 2^0, col 31 -> 2^31)
    shifts = (1 << torch.arange(32, dtype=torch.int64, device=device))

    # Aggregate into 32-bit integers and cast to torch.uint32
    packed_i64 = (bits_grouped * shifts).sum(dim=-1)
    packed_u32 = packed_i64.to(torch.uint32).contiguous()

    # Record metadata directly on the tensor
    packed_u32.orig_in_features = n_cols
    packed_u32.orig_shape = (n_rows, n_cols)

    if return_orig_dim:
        return packed_u32, n_cols
    return packed_u32


class PackedLittleBitLayer(nn.Module):
    """
    PyTorch container module representing a bit-packed LittleBit layer.
    Supports both attribute access (.V_packed) and dictionary access (['V_packed']).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        split_dim: int,
        residual: bool,
        tensors: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_dim = split_dim
        self.residual = residual
        self.has_bias = "bias" in tensors and tensors["bias"] is not None

        # Dimension keys stored as attributes; register remaining tensors as buffers
        dim_keys = {"in_features", "out_features", "split_dim"}
        for name, tensor in tensors.items():
            if name in dim_keys:
                continue
            if tensor is not None:
                self.register_buffer(name, tensor)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in PackedLittleBitLayer")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return [k for k, _ in self.named_buffers()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import lazily to avoid circular dependency
        from kernel_ref import littlebit_forward_ref
        return littlebit_forward_ref(x, self)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"split_dim={self.split_dim}, residual={self.residual}, has_bias={self.has_bias}"
        )


def export_layer_to_safetensors(
    layer: Any,
    file_path: str,
) -> Dict[str, Any]:
    """
    Extracts binary weights and scale vectors from a LittleBitLinear layer,
    packs the {-1, +1} matrices into uint32 tensors, casts scales to FP16,
    and saves them to a safetensors file with full metadata.
    
    Parameters
    ----------
    layer : LittleBitLinear or nn.Module or dict
        The source layer containing LittleBit weights and scales.
    file_path : str
        Target file path for the safetensors file.

    Returns
    -------
    dict
        Dictionary of exported tensors and metadata summary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # 1. Inspect layer attributes
    if isinstance(layer, dict):
        d = layer
        in_features = int(d.get("in_features", d["v2"].shape[-1]))
        out_features = int(d.get("out_features", d["u1"].shape[-1]))
        split_dim = int(d.get("split_dim", d["v1"].shape[-1]))
        residual = bool(d.get("residual", "V_R" in d or "V_R_packed" in d))
        has_bias = "bias" in d and d["bias"] is not None

        def get_tensor(k):
            return d.get(k)
        def quantize_val(t):
            return torch.where(t >= 0, torch.tensor(1.0, device=t.device), torch.tensor(-1.0, device=t.device))
    else:
        in_features = layer.in_features
        out_features = layer.out_features
        split_dim = getattr(layer, "split_dim", layer.v1.shape[-1])
        residual = getattr(layer, "residual", False)
        has_bias = getattr(layer, "bias", None) is not None

        def get_tensor(k):
            return getattr(layer, k, None)
        quant_func = getattr(layer, "quantize", lambda t: torch.where(t >= 0, torch.tensor(1.0, device=t.device), torch.tensor(-1.0, device=t.device)))
        def quantize_val(t):
            return quant_func(t.data if hasattr(t, "data") else t)

    # 2. Extract and binarize main branch weights
    V_raw = get_tensor("V")
    U_raw = get_tensor("U")
    if V_raw is None or U_raw is None:
        raise ValueError("Layer missing required weights 'V' or 'U'")

    V_bin = quantize_val(V_raw)
    U_bin = quantize_val(U_raw)

    V_packed = pack_binary_matrix(V_bin)
    U_packed = pack_binary_matrix(U_bin)

    # Scales to FP16
    v2 = get_tensor("v2").to(torch.float16).contiguous()
    v1 = get_tensor("v1").to(torch.float16).contiguous()
    u2 = get_tensor("u2").to(torch.float16).contiguous()
    u1 = get_tensor("u1").to(torch.float16).contiguous()

    tensors_to_save: Dict[str, torch.Tensor] = {
        "V_packed": V_packed,
        "U_packed": U_packed,
        "v2": v2,
        "v1": v1,
        "u2": u2,
        "u1": u1,
        # Shape tensors for direct tensor inspection
        "in_features": torch.tensor(in_features, dtype=torch.int64),
        "out_features": torch.tensor(out_features, dtype=torch.int64),
        "split_dim": torch.tensor(split_dim, dtype=torch.int64),
        "V_shape": torch.tensor([split_dim, in_features], dtype=torch.int64),
        "U_shape": torch.tensor([out_features, split_dim], dtype=torch.int64),
    }

    # 3. Residual branch if present
    if residual:
        V_R_raw = get_tensor("V_R")
        U_R_raw = get_tensor("U_R")
        if V_R_raw is not None and U_R_raw is not None:
            V_R_bin = quantize_val(V_R_raw)
            U_R_bin = quantize_val(U_R_raw)
            tensors_to_save["V_R_packed"] = pack_binary_matrix(V_R_bin)
            tensors_to_save["U_R_packed"] = pack_binary_matrix(U_R_bin)
            tensors_to_save["v2_R"] = get_tensor("v2_R").to(torch.float16).contiguous()
            tensors_to_save["v1_R"] = get_tensor("v1_R").to(torch.float16).contiguous()
            tensors_to_save["u2_R"] = get_tensor("u2_R").to(torch.float16).contiguous()
            tensors_to_save["u1_R"] = get_tensor("u1_R").to(torch.float16).contiguous()
            tensors_to_save["V_R_shape"] = torch.tensor([split_dim, in_features], dtype=torch.int64)
            tensors_to_save["U_R_shape"] = torch.tensor([out_features, split_dim], dtype=torch.int64)

    # 4. Bias if present
    if has_bias:
        bias_raw = get_tensor("bias")
        if bias_raw is not None:
            tensors_to_save["bias"] = bias_raw.to(torch.float16).contiguous()

    # 5. Metadata dictionary for safetensors header
    metadata = {
        "format": "littlebit_bitpacked_v1",
        "in_features": str(in_features),
        "out_features": str(out_features),
        "split_dim": str(split_dim),
        "residual": str(residual),
        "has_bias": str(has_bias),
        "bit_order": "lsb_first",
        "pack_dtype": "uint32",
    }

    save_file(tensors_to_save, file_path, metadata=metadata)
    return {"file_path": file_path, "metadata": metadata, "tensor_keys": list(tensors_to_save.keys())}


def load_packed_layer(
    file_path: str,
    device: str = "cpu",
) -> PackedLittleBitLayer:
    """
    Loads a bit-packed LittleBit layer from a safetensors file.
    
    Parameters
    ----------
    file_path : str
        Path to the .safetensors file.
    device : str, default 'cpu'
        Target device for tensors.

    Returns
    -------
    PackedLittleBitLayer
        PyTorch container with all packed weights and FP16 scales loaded.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read metadata header
    with safe_open(file_path, framework="pt", device=device) as f:
        meta = f.metadata() or {}

    tensors = load_file(file_path, device=device)

    # Parse dimensions from metadata or shape tensors
    if "in_features" in meta:
        in_features = int(meta["in_features"])
        out_features = int(meta["out_features"])
        split_dim = int(meta["split_dim"])
        residual = meta.get("residual", "False").lower() in ("true", "1", "yes")
    else:
        in_features = int(tensors["in_features"].item())
        out_features = int(tensors["out_features"].item())
        split_dim = int(tensors["split_dim"].item())
        residual = "V_R_packed" in tensors

    return PackedLittleBitLayer(
        in_features=in_features,
        out_features=out_features,
        split_dim=split_dim,
        residual=residual,
        tensors=tensors,
    )
