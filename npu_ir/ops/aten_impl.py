"""ATen operator implementations for IR execution."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .registry import register_executor

# Type alias for cleaner signatures
TensorList = List[torch.Tensor]


def _get_tensor(inputs: TensorList, idx: int) -> Optional[torch.Tensor]:
    """Safely get a tensor from inputs list."""
    if idx < len(inputs):
        return inputs[idx]
    return None


def _to_tuple(val: Any, ndim: int = 2) -> Tuple[int, ...]:
    """Convert value to tuple of specified dimensions."""
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val,) * ndim


# ============================================================================
# Convolution Operations
# ============================================================================


@register_executor("aten.conv2d.default")
@register_executor("aten.conv2d")
def execute_conv2d(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute conv2d operation."""
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None

    stride = attrs.get("stride", (1, 1))
    padding = attrs.get("padding", (0, 0))
    dilation = attrs.get("dilation", (1, 1))
    groups = attrs.get("groups", 1)

    result = F.conv2d(
        x,
        weight,
        bias=bias,
        stride=_to_tuple(stride),
        padding=_to_tuple(padding),
        dilation=_to_tuple(dilation),
        groups=groups,
    )
    return [result]


@register_executor("aten.convolution.default")
@register_executor("aten.convolution")
def execute_convolution(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute convolution operation (general)."""
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None

    stride = attrs.get("stride", (1, 1))
    padding = attrs.get("padding", (0, 0))
    dilation = attrs.get("dilation", (1, 1))
    groups = attrs.get("groups", 1)
    transposed = attrs.get("transposed", False)
    output_padding = attrs.get("output_padding", (0, 0))

    result = torch.convolution(
        x,
        weight,
        bias,
        stride=list(stride) if isinstance(stride, tuple) else stride,
        padding=list(padding) if isinstance(padding, tuple) else padding,
        dilation=list(dilation) if isinstance(dilation, tuple) else dilation,
        transposed=transposed,
        output_padding=list(output_padding)
        if isinstance(output_padding, tuple)
        else output_padding,
        groups=groups,
    )
    return [result]


@register_executor("aten.conv1d.default")
@register_executor("aten.conv1d")
def execute_conv1d(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute conv1d operation."""
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None

    stride = attrs.get("stride", 1)
    padding = attrs.get("padding", 0)
    dilation = attrs.get("dilation", 1)
    groups = attrs.get("groups", 1)

    result = F.conv1d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    return [result]


# ============================================================================
# Linear/Matrix Operations
# ============================================================================


@register_executor("aten.linear.default")
@register_executor("aten.linear")
def execute_linear(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute linear operation."""
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) > 2 else None

    result = F.linear(x, weight, bias)
    return [result]


@register_executor("aten.addmm.default")
@register_executor("aten.addmm")
def execute_addmm(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute addmm operation."""
    bias = inputs[0]
    x = inputs[1]
    weight = inputs[2]

    result = torch.addmm(bias, x, weight)
    return [result]


@register_executor("aten.mm.default")
@register_executor("aten.mm")
def execute_mm(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute matrix multiplication."""
    result = torch.mm(inputs[0], inputs[1])
    return [result]


@register_executor("aten.bmm.default")
@register_executor("aten.bmm")
def execute_bmm(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute batched matrix multiplication."""
    result = torch.bmm(inputs[0], inputs[1])
    return [result]


@register_executor("aten.matmul.default")
@register_executor("aten.matmul")
def execute_matmul(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute matmul operation."""
    result = torch.matmul(inputs[0], inputs[1])
    return [result]


# ============================================================================
# Activation Functions
# ============================================================================


@register_executor("aten.relu.default")
@register_executor("aten.relu")
@register_executor("aten.relu_")
def execute_relu(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute ReLU operation."""
    return [F.relu(inputs[0])]


@register_executor("aten.gelu.default")
@register_executor("aten.gelu")
def execute_gelu(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute GELU operation."""
    approximate = attrs.get("approximate", "none")
    return [F.gelu(inputs[0], approximate=approximate)]


@register_executor("aten.silu.default")
@register_executor("aten.silu")
def execute_silu(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute SiLU/Swish operation."""
    return [F.silu(inputs[0])]


@register_executor("aten.sigmoid.default")
@register_executor("aten.sigmoid")
def execute_sigmoid(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute sigmoid operation."""
    return [torch.sigmoid(inputs[0])]


@register_executor("aten.tanh.default")
@register_executor("aten.tanh")
def execute_tanh(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute tanh operation."""
    return [torch.tanh(inputs[0])]


@register_executor("aten.leaky_relu.default")
@register_executor("aten.leaky_relu")
def execute_leaky_relu(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute leaky ReLU operation."""
    negative_slope = attrs.get("negative_slope", 0.01)
    return [F.leaky_relu(inputs[0], negative_slope=negative_slope)]


@register_executor("aten.hardswish.default")
@register_executor("aten.hardswish")
def execute_hardswish(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute hardswish operation."""
    return [F.hardswish(inputs[0])]


@register_executor("aten.hardsigmoid.default")
@register_executor("aten.hardsigmoid")
def execute_hardsigmoid(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute hardsigmoid operation."""
    return [F.hardsigmoid(inputs[0])]


# ============================================================================
# Normalization Operations
# ============================================================================


@register_executor("aten.batch_norm.default")
@register_executor("aten.batch_norm")
@register_executor("aten._native_batch_norm_legit_no_training.default")
def execute_batch_norm(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute batch normalization operation."""
    x = inputs[0]
    weight = inputs[1] if len(inputs) > 1 else None
    bias = inputs[2] if len(inputs) > 2 else None
    running_mean = inputs[3] if len(inputs) > 3 else None
    running_var = inputs[4] if len(inputs) > 4 else None

    training = attrs.get("training", False)
    momentum = attrs.get("momentum", 0.1)
    eps = attrs.get("eps", 1e-5)

    result = F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )
    return [result]


@register_executor("aten.layer_norm.default")
@register_executor("aten.layer_norm")
@register_executor("aten.native_layer_norm.default")
def execute_layer_norm(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute layer normalization operation."""
    x = inputs[0]
    weight = inputs[1] if len(inputs) > 1 else None
    bias = inputs[2] if len(inputs) > 2 else None

    normalized_shape = attrs.get("normalized_shape", x.shape[-1:])
    eps = attrs.get("eps", 1e-5)

    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    result = F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
    return [result]


@register_executor("aten.group_norm.default")
@register_executor("aten.group_norm")
def execute_group_norm(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute group normalization operation."""
    x = inputs[0]
    weight = inputs[1] if len(inputs) > 1 else None
    bias = inputs[2] if len(inputs) > 2 else None

    num_groups = attrs.get("num_groups", 1)
    eps = attrs.get("eps", 1e-5)

    result = F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)
    return [result]


# ============================================================================
# Pooling Operations
# ============================================================================


@register_executor("aten.max_pool2d.default")
@register_executor("aten.max_pool2d")
@register_executor("aten.max_pool2d_with_indices.default")
def execute_max_pool2d(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute max pooling 2D operation."""
    kernel_size = attrs.get("kernel_size", (2, 2))
    stride = attrs.get("stride", kernel_size)
    padding = attrs.get("padding", (0, 0))
    dilation = attrs.get("dilation", (1, 1))
    ceil_mode = attrs.get("ceil_mode", False)

    result = F.max_pool2d(
        inputs[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    return [result]


@register_executor("aten.avg_pool2d.default")
@register_executor("aten.avg_pool2d")
def execute_avg_pool2d(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute average pooling 2D operation."""
    kernel_size = attrs.get("kernel_size", (2, 2))
    stride = attrs.get("stride", kernel_size)
    padding = attrs.get("padding", (0, 0))
    ceil_mode = attrs.get("ceil_mode", False)
    count_include_pad = attrs.get("count_include_pad", True)

    result = F.avg_pool2d(
        inputs[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    return [result]


@register_executor("aten.adaptive_avg_pool2d.default")
@register_executor("aten.adaptive_avg_pool2d")
@register_executor("aten._adaptive_avg_pool2d.default")
def execute_adaptive_avg_pool2d(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute adaptive average pooling 2D operation."""
    output_size = attrs.get("output_size", (1, 1))
    result = F.adaptive_avg_pool2d(inputs[0], output_size)
    return [result]


# ============================================================================
# Element-wise Operations
# ============================================================================


@register_executor("aten.add.Tensor")
@register_executor("aten.add.default")
@register_executor("aten.add")
@register_executor("aten.add_.Tensor")
@register_executor("aten.add_.default")
@register_executor("aten.add_")
def execute_add(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute element-wise addition operation."""
    alpha = attrs.get("alpha", 1)
    if len(inputs) < 2:
        # Scalar add - second operand might be in attrs
        other = attrs.get("other", 0)
        return [torch.add(inputs[0], other, alpha=alpha)]
    return [torch.add(inputs[0], inputs[1], alpha=alpha)]


@register_executor("aten.sub.Tensor")
@register_executor("aten.sub.default")
@register_executor("aten.sub")
def execute_sub(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute element-wise subtraction operation."""
    alpha = attrs.get("alpha", 1)
    if len(inputs) < 2:
        other = attrs.get("other", 0)
        return [torch.sub(inputs[0], other, alpha=alpha)]
    return [torch.sub(inputs[0], inputs[1], alpha=alpha)]


@register_executor("aten.mul.Tensor")
@register_executor("aten.mul.default")
@register_executor("aten.mul")
def execute_mul(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute element-wise multiplication operation."""
    if len(inputs) < 2:
        other = attrs.get("other", 1)
        return [torch.mul(inputs[0], other)]
    return [torch.mul(inputs[0], inputs[1])]


@register_executor("aten.div.Tensor")
@register_executor("aten.div.default")
@register_executor("aten.div")
def execute_div(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute element-wise division operation."""
    if len(inputs) < 2:
        other = attrs.get("other", 1)
        return [torch.div(inputs[0], other)]
    return [torch.div(inputs[0], inputs[1])]


@register_executor("aten.pow.Tensor_Scalar")
@register_executor("aten.pow")
def execute_pow(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute power operation."""
    exponent = attrs.get("exponent", 2) if len(inputs) < 2 else inputs[1]
    return [torch.pow(inputs[0], exponent)]


@register_executor("aten.sqrt.default")
@register_executor("aten.sqrt")
def execute_sqrt(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute square root operation."""
    return [torch.sqrt(inputs[0])]


@register_executor("aten.rsqrt.default")
@register_executor("aten.rsqrt")
def execute_rsqrt(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute reciprocal square root operation."""
    return [torch.rsqrt(inputs[0])]


@register_executor("aten.neg.default")
@register_executor("aten.neg")
def execute_neg(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute negation operation."""
    return [torch.neg(inputs[0])]


@register_executor("aten.abs.default")
@register_executor("aten.abs")
def execute_abs(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute absolute value operation."""
    return [torch.abs(inputs[0])]


@register_executor("aten.exp.default")
@register_executor("aten.exp")
def execute_exp(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute exponential operation."""
    return [torch.exp(inputs[0])]


@register_executor("aten.log.default")
@register_executor("aten.log")
def execute_log(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute logarithm operation."""
    return [torch.log(inputs[0])]


@register_executor("aten.clamp.default")
@register_executor("aten.clamp")
def execute_clamp(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute clamp operation."""
    min_val = attrs.get("min", None)
    max_val = attrs.get("max", None)
    return [torch.clamp(inputs[0], min=min_val, max=max_val)]


# ============================================================================
# Shape Operations
# ============================================================================


@register_executor("aten.view.default")
@register_executor("aten.view")
@register_executor("aten._unsafe_view.default")
def execute_view(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute view/reshape operation."""
    shape = attrs.get("shape", attrs.get("size", [-1]))
    return [inputs[0].view(*shape)]


@register_executor("aten.reshape.default")
@register_executor("aten.reshape")
def execute_reshape(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute reshape operation."""
    shape = attrs.get("shape", attrs.get("size", [-1]))
    return [inputs[0].reshape(*shape)]


@register_executor("aten.permute.default")
@register_executor("aten.permute")
def execute_permute(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute permute operation."""
    dims = attrs.get("dims", list(range(inputs[0].dim())))
    return [inputs[0].permute(*dims)]


@register_executor("aten.transpose.int")
@register_executor("aten.transpose.default")
@register_executor("aten.transpose")
def execute_transpose(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute transpose operation."""
    dim0 = attrs.get("dim0", 0)
    dim1 = attrs.get("dim1", 1)
    return [inputs[0].transpose(dim0, dim1)]


@register_executor("aten.t.default")
def execute_t(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute 2D transpose operation."""
    return [inputs[0].t()]


@register_executor("aten.flatten.using_ints")
@register_executor("aten.flatten.default")
@register_executor("aten.flatten")
def execute_flatten(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute flatten operation."""
    start_dim = attrs.get("start_dim", 0)
    end_dim = attrs.get("end_dim", -1)
    return [inputs[0].flatten(start_dim, end_dim)]


@register_executor("aten.squeeze.dim")
@register_executor("aten.squeeze.default")
@register_executor("aten.squeeze")
def execute_squeeze(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute squeeze operation."""
    dim = attrs.get("dim", None)
    if dim is not None:
        return [inputs[0].squeeze(dim)]
    return [inputs[0].squeeze()]


@register_executor("aten.unsqueeze.default")
@register_executor("aten.unsqueeze")
def execute_unsqueeze(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute unsqueeze operation."""
    dim = attrs.get("dim", 0)
    return [inputs[0].unsqueeze(dim)]


@register_executor("aten.expand.default")
@register_executor("aten.expand")
def execute_expand(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute expand operation."""
    size = attrs.get("size", inputs[0].shape)
    return [inputs[0].expand(*size)]


# ============================================================================
# Concatenation/Split Operations
# ============================================================================


@register_executor("aten.cat.default")
@register_executor("aten.cat")
def execute_cat(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute concatenation operation."""
    dim = attrs.get("dim", 0)
    return [torch.cat(inputs, dim=dim)]


@register_executor("aten.split.Tensor")
@register_executor("aten.split.default")
@register_executor("aten.split")
def execute_split(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute split operation."""
    split_size = attrs.get("split_size", 1)
    dim = attrs.get("dim", 0)
    return list(torch.split(inputs[0], split_size, dim=dim))


@register_executor("aten.stack.default")
@register_executor("aten.stack")
def execute_stack(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute stack operation."""
    dim = attrs.get("dim", 0)
    return [torch.stack(inputs, dim=dim)]


# ============================================================================
# Reduction Operations
# ============================================================================


@register_executor("aten.mean.dim")
@register_executor("aten.mean.default")
@register_executor("aten.mean")
def execute_mean(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute mean reduction operation."""
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return [inputs[0].mean(dim=dim, keepdim=keepdim)]
    return [inputs[0].mean()]


@register_executor("aten.sum.dim_IntList")
@register_executor("aten.sum.default")
@register_executor("aten.sum")
def execute_sum(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute sum reduction operation."""
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        return [inputs[0].sum(dim=dim, keepdim=keepdim)]
    return [inputs[0].sum()]


@register_executor("aten.max.dim")
@register_executor("aten.max.default")
@register_executor("aten.max")
@register_executor("aten.amax.default")
def execute_max(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute max reduction operation."""
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        result = inputs[0].amax(dim=dim, keepdim=keepdim)
    else:
        result = inputs[0].max()
    return [result]


@register_executor("aten.min.dim")
@register_executor("aten.min.default")
@register_executor("aten.min")
@register_executor("aten.amin.default")
def execute_min(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute min reduction operation."""
    dim = attrs.get("dim", None)
    keepdim = attrs.get("keepdim", False)
    if dim is not None:
        result = inputs[0].amin(dim=dim, keepdim=keepdim)
    else:
        result = inputs[0].min()
    return [result]


# ============================================================================
# Softmax/Attention Operations
# ============================================================================


@register_executor("aten.softmax.int")
@register_executor("aten.softmax.default")
@register_executor("aten.softmax")
@register_executor("aten._softmax.default")
def execute_softmax(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute softmax operation."""
    dim = attrs.get("dim", -1)
    return [F.softmax(inputs[0], dim=dim)]


@register_executor("aten.log_softmax.int")
@register_executor("aten.log_softmax.default")
@register_executor("aten.log_softmax")
@register_executor("aten._log_softmax.default")
def execute_log_softmax(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute log softmax operation."""
    dim = attrs.get("dim", -1)
    return [F.log_softmax(inputs[0], dim=dim)]


@register_executor("aten.scaled_dot_product_attention.default")
@register_executor("aten.scaled_dot_product_attention")
def execute_scaled_dot_product_attention(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute scaled dot product attention operation."""
    q, k, v = inputs[0], inputs[1], inputs[2]
    attn_mask = inputs[3] if len(inputs) > 3 else None
    dropout_p = attrs.get("dropout_p", 0.0)
    is_causal = attrs.get("is_causal", False)
    scale = attrs.get("scale", None)

    result = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
    return [result]


# ============================================================================
# Embedding Operations
# ============================================================================


@register_executor("aten.embedding.default")
@register_executor("aten.embedding")
def execute_embedding(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute embedding operation."""
    weight = inputs[0]
    indices = inputs[1]
    padding_idx = attrs.get("padding_idx", None)
    return [F.embedding(indices, weight, padding_idx=padding_idx)]


# ============================================================================
# Dropout Operations
# ============================================================================


@register_executor("aten.dropout.default")
@register_executor("aten.dropout")
@register_executor("aten.native_dropout.default")
def execute_dropout(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute dropout operation (identity in eval mode)."""
    # In eval mode, dropout is identity
    training = attrs.get("training", False)
    p = attrs.get("p", 0.5)
    if training:
        return [F.dropout(inputs[0], p=p, training=True)]
    return [inputs[0]]


# ============================================================================
# Type Conversion Operations
# ============================================================================


@register_executor("aten.to.dtype")
@register_executor("aten.to.device")
@register_executor("aten.to.dtype_layout")
@register_executor("aten._to_copy.default")
def execute_to(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute type/device conversion operation."""
    dtype = attrs.get("dtype", None)
    device = attrs.get("device", None)
    if dtype is not None:
        return [inputs[0].to(dtype=dtype)]
    if device is not None:
        return [inputs[0].to(device=device)]
    return [inputs[0]]


@register_executor("aten.contiguous.default")
@register_executor("aten.contiguous")
def execute_contiguous(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute contiguous operation."""
    return [inputs[0].contiguous()]


@register_executor("aten.clone.default")
@register_executor("aten.clone")
def execute_clone(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute clone operation."""
    return [inputs[0].clone()]


# ============================================================================
# Indexing Operations
# ============================================================================


@register_executor("aten.select.int")
@register_executor("aten.select.default")
@register_executor("aten.select")
def execute_select(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute select operation."""
    dim = attrs.get("dim", 0)
    index = attrs.get("index", 0)
    return [inputs[0].select(dim, index)]


@register_executor("aten.slice.Tensor")
@register_executor("aten.slice.default")
@register_executor("aten.slice")
def execute_slice(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute slice operation."""
    dim = attrs.get("dim", 0)
    start = attrs.get("start", None)
    end = attrs.get("end", None)
    step = attrs.get("step", 1)

    # Build slice object
    slices = [slice(None)] * inputs[0].dim()
    slices[dim] = slice(start, end, step)
    return [inputs[0][tuple(slices)]]


@register_executor("aten.gather.default")
@register_executor("aten.gather")
def execute_gather(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute gather operation."""
    dim = attrs.get("dim", 0)
    return [torch.gather(inputs[0], dim, inputs[1])]


# ============================================================================
# Comparison Operations
# ============================================================================


@register_executor("aten.eq.Tensor")
@register_executor("aten.eq.Scalar")
@register_executor("aten.eq")
def execute_eq(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute equality comparison operation."""
    other = inputs[1] if len(inputs) > 1 else attrs.get("other", 0)
    return [torch.eq(inputs[0], other)]


@register_executor("aten.ne.Tensor")
@register_executor("aten.ne.Scalar")
@register_executor("aten.ne")
def execute_ne(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute not-equal comparison operation."""
    other = inputs[1] if len(inputs) > 1 else attrs.get("other", 0)
    return [torch.ne(inputs[0], other)]


@register_executor("aten.lt.Tensor")
@register_executor("aten.lt.Scalar")
@register_executor("aten.lt")
def execute_lt(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute less-than comparison operation."""
    other = inputs[1] if len(inputs) > 1 else attrs.get("other", 0)
    return [torch.lt(inputs[0], other)]


@register_executor("aten.gt.Tensor")
@register_executor("aten.gt.Scalar")
@register_executor("aten.gt")
def execute_gt(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute greater-than comparison operation."""
    other = inputs[1] if len(inputs) > 1 else attrs.get("other", 0)
    return [torch.gt(inputs[0], other)]


@register_executor("aten.where.self")
@register_executor("aten.where.default")
@register_executor("aten.where")
def execute_where(inputs: TensorList, attrs: Dict[str, Any]) -> TensorList:
    """Execute where operation."""
    return [torch.where(inputs[0], inputs[1], inputs[2])]
