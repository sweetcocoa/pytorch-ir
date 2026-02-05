"""ATen operator mappings for IR conversion."""

from typing import Dict, Any, List, Optional
from ..ir import OpNode, TensorMeta
from ..analyzer import NodeInfo
from .registry import register_op


def _normalize_op_type(target: Any) -> str:
    """Normalize the operator type to a standard string format."""
    target_str = str(target)

    # Handle torch.ops.aten.* format
    if "torch.ops.aten." in target_str:
        # Extract the operation name
        parts = target_str.replace("torch.ops.aten.", "aten.").split()
        return parts[0] if parts else target_str

    # Handle aten::* format
    if "aten::" in target_str:
        return target_str.replace("aten::", "aten.")

    return target_str


def _create_op_node(
    node_info: NodeInfo,
    op_type: Optional[str] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> OpNode:
    """Helper to create an OpNode from NodeInfo."""
    final_op_type = op_type or _normalize_op_type(node_info.target)

    attrs = dict(node_info.attrs)
    if extra_attrs:
        attrs.update(extra_attrs)

    return OpNode(
        name=node_info.name,
        op_type=final_op_type,
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=attrs,
    )


# ============================================================================
# Convolution Operations
# ============================================================================


@register_op("aten.conv2d.default")
@register_op("aten.conv2d")
@register_op("aten.convolution.default")
@register_op("aten.convolution")
@register_op("aten._conv_depthwise2d")
def convert_conv2d(node_info: NodeInfo) -> OpNode:
    """Convert conv2d operation."""
    return _create_op_node(node_info)


@register_op("aten.conv1d.default")
@register_op("aten.conv1d")
def convert_conv1d(node_info: NodeInfo) -> OpNode:
    """Convert conv1d operation."""
    return _create_op_node(node_info)


@register_op("aten.conv3d.default")
@register_op("aten.conv3d")
def convert_conv3d(node_info: NodeInfo) -> OpNode:
    """Convert conv3d operation."""
    return _create_op_node(node_info)


# ============================================================================
# Linear/Matrix Operations
# ============================================================================


@register_op("aten.linear.default")
@register_op("aten.linear")
def convert_linear(node_info: NodeInfo) -> OpNode:
    """Convert linear operation."""
    return _create_op_node(node_info)


@register_op("aten.addmm.default")
@register_op("aten.addmm")
def convert_addmm(node_info: NodeInfo) -> OpNode:
    """Convert addmm operation (used by linear layers)."""
    return _create_op_node(node_info)


@register_op("aten.mm.default")
@register_op("aten.mm")
def convert_mm(node_info: NodeInfo) -> OpNode:
    """Convert matrix multiplication operation."""
    return _create_op_node(node_info)


@register_op("aten.bmm.default")
@register_op("aten.bmm")
def convert_bmm(node_info: NodeInfo) -> OpNode:
    """Convert batched matrix multiplication operation."""
    return _create_op_node(node_info)


@register_op("aten.matmul.default")
@register_op("aten.matmul")
def convert_matmul(node_info: NodeInfo) -> OpNode:
    """Convert matmul operation."""
    return _create_op_node(node_info)


# ============================================================================
# Activation Functions
# ============================================================================


@register_op("aten.relu.default")
@register_op("aten.relu")
@register_op("aten.relu_")
def convert_relu(node_info: NodeInfo) -> OpNode:
    """Convert ReLU operation."""
    return _create_op_node(node_info)


@register_op("aten.gelu.default")
@register_op("aten.gelu")
def convert_gelu(node_info: NodeInfo) -> OpNode:
    """Convert GELU operation."""
    return _create_op_node(node_info)


@register_op("aten.silu.default")
@register_op("aten.silu")
def convert_silu(node_info: NodeInfo) -> OpNode:
    """Convert SiLU/Swish operation."""
    return _create_op_node(node_info)


@register_op("aten.sigmoid.default")
@register_op("aten.sigmoid")
def convert_sigmoid(node_info: NodeInfo) -> OpNode:
    """Convert sigmoid operation."""
    return _create_op_node(node_info)


@register_op("aten.tanh.default")
@register_op("aten.tanh")
def convert_tanh(node_info: NodeInfo) -> OpNode:
    """Convert tanh operation."""
    return _create_op_node(node_info)


@register_op("aten.leaky_relu.default")
@register_op("aten.leaky_relu")
def convert_leaky_relu(node_info: NodeInfo) -> OpNode:
    """Convert leaky ReLU operation."""
    return _create_op_node(node_info)


@register_op("aten.hardswish.default")
@register_op("aten.hardswish")
def convert_hardswish(node_info: NodeInfo) -> OpNode:
    """Convert hardswish operation."""
    return _create_op_node(node_info)


@register_op("aten.hardsigmoid.default")
@register_op("aten.hardsigmoid")
def convert_hardsigmoid(node_info: NodeInfo) -> OpNode:
    """Convert hardsigmoid operation."""
    return _create_op_node(node_info)


# ============================================================================
# Normalization Operations
# ============================================================================


@register_op("aten.batch_norm.default")
@register_op("aten.batch_norm")
@register_op("aten._native_batch_norm_legit_no_training.default")
@register_op("aten._native_batch_norm_legit.default")
def convert_batch_norm(node_info: NodeInfo) -> OpNode:
    """Convert batch normalization operation."""
    return _create_op_node(node_info)


@register_op("aten.layer_norm.default")
@register_op("aten.layer_norm")
@register_op("aten.native_layer_norm.default")
def convert_layer_norm(node_info: NodeInfo) -> OpNode:
    """Convert layer normalization operation."""
    return _create_op_node(node_info)


@register_op("aten.group_norm.default")
@register_op("aten.group_norm")
def convert_group_norm(node_info: NodeInfo) -> OpNode:
    """Convert group normalization operation."""
    return _create_op_node(node_info)


@register_op("aten.instance_norm.default")
@register_op("aten.instance_norm")
def convert_instance_norm(node_info: NodeInfo) -> OpNode:
    """Convert instance normalization operation."""
    return _create_op_node(node_info)


# ============================================================================
# Pooling Operations
# ============================================================================


@register_op("aten.max_pool2d.default")
@register_op("aten.max_pool2d")
@register_op("aten.max_pool2d_with_indices.default")
def convert_max_pool2d(node_info: NodeInfo) -> OpNode:
    """Convert max pooling 2D operation."""
    return _create_op_node(node_info)


@register_op("aten.avg_pool2d.default")
@register_op("aten.avg_pool2d")
def convert_avg_pool2d(node_info: NodeInfo) -> OpNode:
    """Convert average pooling 2D operation."""
    return _create_op_node(node_info)


@register_op("aten.adaptive_avg_pool2d.default")
@register_op("aten.adaptive_avg_pool2d")
@register_op("aten._adaptive_avg_pool2d.default")
def convert_adaptive_avg_pool2d(node_info: NodeInfo) -> OpNode:
    """Convert adaptive average pooling 2D operation."""
    return _create_op_node(node_info)


@register_op("aten.adaptive_max_pool2d.default")
@register_op("aten.adaptive_max_pool2d")
def convert_adaptive_max_pool2d(node_info: NodeInfo) -> OpNode:
    """Convert adaptive max pooling 2D operation."""
    return _create_op_node(node_info)


# ============================================================================
# Element-wise Operations
# ============================================================================


@register_op("aten.add.Tensor")
@register_op("aten.add.default")
@register_op("aten.add")
@register_op("aten.add_.Tensor")
@register_op("aten.add_.default")
@register_op("aten.add_")
def convert_add(node_info: NodeInfo) -> OpNode:
    """Convert element-wise addition operation."""
    return _create_op_node(node_info)


@register_op("aten.sub.Tensor")
@register_op("aten.sub.default")
@register_op("aten.sub")
def convert_sub(node_info: NodeInfo) -> OpNode:
    """Convert element-wise subtraction operation."""
    return _create_op_node(node_info)


@register_op("aten.mul.Tensor")
@register_op("aten.mul.default")
@register_op("aten.mul")
def convert_mul(node_info: NodeInfo) -> OpNode:
    """Convert element-wise multiplication operation."""
    return _create_op_node(node_info)


@register_op("aten.div.Tensor")
@register_op("aten.div.default")
@register_op("aten.div")
def convert_div(node_info: NodeInfo) -> OpNode:
    """Convert element-wise division operation."""
    return _create_op_node(node_info)


@register_op("aten.pow.Tensor_Scalar")
@register_op("aten.pow")
def convert_pow(node_info: NodeInfo) -> OpNode:
    """Convert power operation."""
    return _create_op_node(node_info)


@register_op("aten.sqrt.default")
@register_op("aten.sqrt")
def convert_sqrt(node_info: NodeInfo) -> OpNode:
    """Convert square root operation."""
    return _create_op_node(node_info)


@register_op("aten.rsqrt.default")
@register_op("aten.rsqrt")
def convert_rsqrt(node_info: NodeInfo) -> OpNode:
    """Convert reciprocal square root operation."""
    return _create_op_node(node_info)


@register_op("aten.neg.default")
@register_op("aten.neg")
def convert_neg(node_info: NodeInfo) -> OpNode:
    """Convert negation operation."""
    return _create_op_node(node_info)


@register_op("aten.abs.default")
@register_op("aten.abs")
def convert_abs(node_info: NodeInfo) -> OpNode:
    """Convert absolute value operation."""
    return _create_op_node(node_info)


@register_op("aten.exp.default")
@register_op("aten.exp")
def convert_exp(node_info: NodeInfo) -> OpNode:
    """Convert exponential operation."""
    return _create_op_node(node_info)


@register_op("aten.log.default")
@register_op("aten.log")
def convert_log(node_info: NodeInfo) -> OpNode:
    """Convert logarithm operation."""
    return _create_op_node(node_info)


@register_op("aten.clamp.default")
@register_op("aten.clamp")
@register_op("aten.clamp_min.default")
@register_op("aten.clamp_max.default")
def convert_clamp(node_info: NodeInfo) -> OpNode:
    """Convert clamp operation."""
    return _create_op_node(node_info)


# ============================================================================
# Shape Operations
# ============================================================================


@register_op("aten.view.default")
@register_op("aten.view")
@register_op("aten._unsafe_view.default")
def convert_view(node_info: NodeInfo) -> OpNode:
    """Convert view/reshape operation."""
    return _create_op_node(node_info)


@register_op("aten.reshape.default")
@register_op("aten.reshape")
def convert_reshape(node_info: NodeInfo) -> OpNode:
    """Convert reshape operation."""
    return _create_op_node(node_info)


@register_op("aten.permute.default")
@register_op("aten.permute")
def convert_permute(node_info: NodeInfo) -> OpNode:
    """Convert permute operation."""
    return _create_op_node(node_info)


@register_op("aten.transpose.int")
@register_op("aten.transpose.default")
@register_op("aten.transpose")
@register_op("aten.t.default")
def convert_transpose(node_info: NodeInfo) -> OpNode:
    """Convert transpose operation."""
    return _create_op_node(node_info)


@register_op("aten.flatten.using_ints")
@register_op("aten.flatten.default")
@register_op("aten.flatten")
def convert_flatten(node_info: NodeInfo) -> OpNode:
    """Convert flatten operation."""
    return _create_op_node(node_info)


@register_op("aten.squeeze.dim")
@register_op("aten.squeeze.default")
@register_op("aten.squeeze")
def convert_squeeze(node_info: NodeInfo) -> OpNode:
    """Convert squeeze operation."""
    return _create_op_node(node_info)


@register_op("aten.unsqueeze.default")
@register_op("aten.unsqueeze")
def convert_unsqueeze(node_info: NodeInfo) -> OpNode:
    """Convert unsqueeze operation."""
    return _create_op_node(node_info)


@register_op("aten.expand.default")
@register_op("aten.expand")
def convert_expand(node_info: NodeInfo) -> OpNode:
    """Convert expand operation."""
    return _create_op_node(node_info)


@register_op("aten.repeat.default")
@register_op("aten.repeat")
def convert_repeat(node_info: NodeInfo) -> OpNode:
    """Convert repeat operation."""
    return _create_op_node(node_info)


# ============================================================================
# Concatenation/Split Operations
# ============================================================================


@register_op("aten.cat.default")
@register_op("aten.cat")
def convert_cat(node_info: NodeInfo) -> OpNode:
    """Convert concatenation operation."""
    return _create_op_node(node_info)


@register_op("aten.split.Tensor")
@register_op("aten.split.default")
@register_op("aten.split")
@register_op("aten.split_with_sizes.default")
def convert_split(node_info: NodeInfo) -> OpNode:
    """Convert split operation."""
    return _create_op_node(node_info)


@register_op("aten.chunk.default")
@register_op("aten.chunk")
def convert_chunk(node_info: NodeInfo) -> OpNode:
    """Convert chunk operation."""
    return _create_op_node(node_info)


@register_op("aten.stack.default")
@register_op("aten.stack")
def convert_stack(node_info: NodeInfo) -> OpNode:
    """Convert stack operation."""
    return _create_op_node(node_info)


# ============================================================================
# Reduction Operations
# ============================================================================


@register_op("aten.mean.dim")
@register_op("aten.mean.default")
@register_op("aten.mean")
def convert_mean(node_info: NodeInfo) -> OpNode:
    """Convert mean reduction operation."""
    return _create_op_node(node_info)


@register_op("aten.sum.dim_IntList")
@register_op("aten.sum.default")
@register_op("aten.sum")
def convert_sum(node_info: NodeInfo) -> OpNode:
    """Convert sum reduction operation."""
    return _create_op_node(node_info)


@register_op("aten.max.dim")
@register_op("aten.max.default")
@register_op("aten.max")
@register_op("aten.amax.default")
def convert_max(node_info: NodeInfo) -> OpNode:
    """Convert max reduction operation."""
    return _create_op_node(node_info)


@register_op("aten.min.dim")
@register_op("aten.min.default")
@register_op("aten.min")
@register_op("aten.amin.default")
def convert_min(node_info: NodeInfo) -> OpNode:
    """Convert min reduction operation."""
    return _create_op_node(node_info)


# ============================================================================
# Softmax/Attention Operations
# ============================================================================


@register_op("aten.softmax.int")
@register_op("aten.softmax.default")
@register_op("aten.softmax")
@register_op("aten._softmax.default")
def convert_softmax(node_info: NodeInfo) -> OpNode:
    """Convert softmax operation."""
    return _create_op_node(node_info)


@register_op("aten.log_softmax.int")
@register_op("aten.log_softmax.default")
@register_op("aten.log_softmax")
@register_op("aten._log_softmax.default")
def convert_log_softmax(node_info: NodeInfo) -> OpNode:
    """Convert log softmax operation."""
    return _create_op_node(node_info)


@register_op("aten.scaled_dot_product_attention.default")
@register_op("aten.scaled_dot_product_attention")
@register_op("aten._scaled_dot_product_flash_attention.default")
@register_op("aten._scaled_dot_product_efficient_attention.default")
def convert_scaled_dot_product_attention(node_info: NodeInfo) -> OpNode:
    """Convert scaled dot product attention operation."""
    return _create_op_node(node_info)


# ============================================================================
# Embedding Operations
# ============================================================================


@register_op("aten.embedding.default")
@register_op("aten.embedding")
def convert_embedding(node_info: NodeInfo) -> OpNode:
    """Convert embedding operation."""
    return _create_op_node(node_info)


# ============================================================================
# Dropout Operations
# ============================================================================


@register_op("aten.dropout.default")
@register_op("aten.dropout")
@register_op("aten.native_dropout.default")
def convert_dropout(node_info: NodeInfo) -> OpNode:
    """Convert dropout operation (identity in eval mode)."""
    return _create_op_node(node_info)


# ============================================================================
# Type Conversion Operations
# ============================================================================


@register_op("aten.to.dtype")
@register_op("aten.to.device")
@register_op("aten.to.dtype_layout")
@register_op("aten._to_copy.default")
def convert_to(node_info: NodeInfo) -> OpNode:
    """Convert type/device conversion operation."""
    return _create_op_node(node_info)


@register_op("aten.contiguous.default")
@register_op("aten.contiguous")
def convert_contiguous(node_info: NodeInfo) -> OpNode:
    """Convert contiguous operation."""
    return _create_op_node(node_info)


@register_op("aten.clone.default")
@register_op("aten.clone")
def convert_clone(node_info: NodeInfo) -> OpNode:
    """Convert clone operation."""
    return _create_op_node(node_info)


# ============================================================================
# Indexing Operations
# ============================================================================


@register_op("aten.select.int")
@register_op("aten.select.default")
@register_op("aten.select")
def convert_select(node_info: NodeInfo) -> OpNode:
    """Convert select operation."""
    return _create_op_node(node_info)


@register_op("aten.slice.Tensor")
@register_op("aten.slice.default")
@register_op("aten.slice")
def convert_slice(node_info: NodeInfo) -> OpNode:
    """Convert slice operation."""
    return _create_op_node(node_info)


@register_op("aten.index.Tensor")
@register_op("aten.index.default")
@register_op("aten.index")
def convert_index(node_info: NodeInfo) -> OpNode:
    """Convert index operation."""
    return _create_op_node(node_info)


@register_op("aten.gather.default")
@register_op("aten.gather")
def convert_gather(node_info: NodeInfo) -> OpNode:
    """Convert gather operation."""
    return _create_op_node(node_info)


@register_op("aten.scatter.value")
@register_op("aten.scatter.default")
@register_op("aten.scatter")
def convert_scatter(node_info: NodeInfo) -> OpNode:
    """Convert scatter operation."""
    return _create_op_node(node_info)


# ============================================================================
# Comparison Operations
# ============================================================================


@register_op("aten.eq.Tensor")
@register_op("aten.eq.Scalar")
@register_op("aten.eq")
def convert_eq(node_info: NodeInfo) -> OpNode:
    """Convert equality comparison operation."""
    return _create_op_node(node_info)


@register_op("aten.ne.Tensor")
@register_op("aten.ne.Scalar")
@register_op("aten.ne")
def convert_ne(node_info: NodeInfo) -> OpNode:
    """Convert not-equal comparison operation."""
    return _create_op_node(node_info)


@register_op("aten.lt.Tensor")
@register_op("aten.lt.Scalar")
@register_op("aten.lt")
def convert_lt(node_info: NodeInfo) -> OpNode:
    """Convert less-than comparison operation."""
    return _create_op_node(node_info)


@register_op("aten.le.Tensor")
@register_op("aten.le.Scalar")
@register_op("aten.le")
def convert_le(node_info: NodeInfo) -> OpNode:
    """Convert less-than-or-equal comparison operation."""
    return _create_op_node(node_info)


@register_op("aten.gt.Tensor")
@register_op("aten.gt.Scalar")
@register_op("aten.gt")
def convert_gt(node_info: NodeInfo) -> OpNode:
    """Convert greater-than comparison operation."""
    return _create_op_node(node_info)


@register_op("aten.ge.Tensor")
@register_op("aten.ge.Scalar")
@register_op("aten.ge")
def convert_ge(node_info: NodeInfo) -> OpNode:
    """Convert greater-than-or-equal comparison operation."""
    return _create_op_node(node_info)


@register_op("aten.where.self")
@register_op("aten.where.default")
@register_op("aten.where")
def convert_where(node_info: NodeInfo) -> OpNode:
    """Convert where operation."""
    return _create_op_node(node_info)


# ============================================================================
# Utility function for getting operation type
# ============================================================================


def get_op_type(target: Any) -> str:
    """Get normalized operation type string from a target."""
    return _normalize_op_type(target)
