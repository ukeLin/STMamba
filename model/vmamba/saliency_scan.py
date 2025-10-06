import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

def traverse_saliency_tensor(tensor):
    """
    SNS算法：空间邻域扫描，保持显著区域的空间连续性
    
    输入:
        tensor: 显著性掩码，形状为 [H, W]
    输出:
        显著区域的索引列表，按照空间连续性排序
    """
    tensor = tensor.squeeze()
    result_index = []
    current_row = 0
    direction = 'left_right'
    while current_row < tensor.size(0):
        # 提取当前行的非零元素及其索引
        non_zero_indices = torch.nonzero(tensor[current_row], as_tuple=False).squeeze()
        # 处理单个元素的情况
        if non_zero_indices.ndim == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)
        
        if direction == 'right_left':
            non_zero_indices = torch.flip(non_zero_indices, dims=[-1])
            
        if len(non_zero_indices) > 0:
            non_zero_index = non_zero_indices + current_row * tensor.size(0)
            result_index.extend(non_zero_index.tolist())
            
            if current_row < tensor.size(0) - 1:  # 确保有下一行
                last_index = non_zero_indices[-1].item()

                # 提取下一行的非零元素及其索引
                next_non_zero_indices = torch.nonzero(tensor[current_row + 1], as_tuple=False).squeeze()
                if next_non_zero_indices.ndim == 0:
                    next_non_zero_indices = next_non_zero_indices.unsqueeze(0)

                if len(next_non_zero_indices) > 0:
                    left_index = next_non_zero_indices[0].item()
                    right_index = next_non_zero_indices[-1].item()

                    # 计算曼哈顿距离
                    left_dist = abs(last_index - left_index)
                    right_dist = abs(last_index - right_index)

                    # 根据距离选择最左或最右的非零元素
                    if left_dist <= right_dist:
                        direction = 'left_right'
                    else:
                        direction = 'right_left'

        current_row += 1
    
    return torch.tensor(result_index, device=tensor.device)

def extract_non_zero_values(tensor):
    """
    提取非零值及其掩码
    """
    mask = tensor != 0
    values = tensor[mask]
    return values, mask

def restore_tensor(original_shape, values, mask):
    """
    根据掩码恢复张量
    """
    B, C, H, W = original_shape
    restored = torch.zeros((B, C, H, W), device=values.device)
    restored[mask] = values
    return restored

def restore_saliency_tensor(original_shape, non_zero_values, non_zero_indexs):
    """
    根据索引恢复张量
    """
    B, C, H, W = original_shape
    flat_restored_tensor = torch.zeros((B, C, H * W), device=non_zero_values.device)
    flat_restored_tensor.scatter_(dim=-1, index=non_zero_indexs, src=non_zero_values)
    restored_tensor = flat_restored_tensor.view(B, C, H, W)
    return restored_tensor

class CrossScanBase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        """
        基础扫描函数
        Args:
            x: 输入特征图，形状为(B, C, H, W)
            saliency_mask: 显著性掩码，形状为(B, 1, H, W)
        Returns:
            xs: 扫描后的特征图，形状为(B, 1, C, H*W)
        """
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        ctx.saliency_mask = saliency_mask
        
        # 如果没有提供显著性掩码，使用默认扫描
        if saliency_mask is None:
            scan_path = torch.arange(H * W, device=x.device)
            x_flat = x.view(B, C, -1)
            x_reordered = x_flat[..., scan_path]
            return x_reordered.unsqueeze(1)
        
        # 初始化输出
        xs = torch.zeros((B, 1, C, H*W), device=x.device)
        
        for b in range(B):
            # 将掩码展平
            gt_flat = saliency_mask[b].reshape(1, -1)  # [1, H*W]
            
            # 创建显著区域和非显著区域的掩码
            salient_mask = gt_flat > 0.5     # [1, H*W]
            non_salient_mask = ~salient_mask  # [1, H*W]
            
            # 计算显著区域和非显著区域的索引
            salient_indices = torch.nonzero(salient_mask[0]).squeeze(-1)  # [S]
            non_salient_indices = torch.nonzero(non_salient_mask[0]).squeeze(-1)  # [NS]
            
            # 处理边缘情况
            if salient_indices.numel() == 0:
                salient_indices = torch.tensor([0], device=x.device)
            if non_salient_indices.numel() == 0:
                non_salient_indices = torch.tensor([0], device=x.device)
            
            # 将特征图展平
            x_flat = x[b].reshape(C, -1)  # [C, H*W]
            
            # 根据调用类的名称决定使用哪种扫描顺序
            caller_name = ctx.__class__.__name__
            
            if caller_name == "CrossScan_Regular":
                # 先显著区域，后非显著区域
                scan_order = torch.cat([salient_indices, non_salient_indices])
            elif caller_name == "CrossScan_Saliency1":
                # 先显著区域（反转），后非显著区域（反转）
                scan_order = torch.cat([torch.flip(salient_indices, [0]), torch.flip(non_salient_indices, [0])])
            elif caller_name == "CrossScan_Saliency2":
                # 先非显著区域，后显著区域
                scan_order = torch.cat([non_salient_indices, salient_indices])
            elif caller_name == "CrossScan_Saliency3":
                # 先非显著区域（反转），后显著区域（反转）
                scan_order = torch.cat([torch.flip(non_salient_indices, [0]), torch.flip(salient_indices, [0])])
            else:
                # 默认：先显著区域，后非显著区域
                scan_order = torch.cat([salient_indices, non_salient_indices])
            
            # 确保扫描顺序的长度与原始特征图一致
            L = H * W
            if scan_order.shape[0] < L:
                # 如果长度不足，使用循环填充
                repeats = (L + scan_order.shape[0] - 1) // scan_order.shape[0]
                scan_order = scan_order.repeat(repeats)[:L]
            elif scan_order.shape[0] > L:
                # 如果长度超出，截断
                scan_order = scan_order[:L]
            
            # 扩展索引以匹配通道维度
            scan_order_expanded = scan_order.unsqueeze(0).expand(C, -1)  # [C, H*W]
            
            # 重排序特征
            xs[b, 0] = torch.gather(x_flat, 1, scan_order_expanded)
        
        return xs
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        B, C, H, W = ctx.shape
        saliency_mask = ctx.saliency_mask
        
        if saliency_mask is None:
            # 默认反向传播
            grad_flat = grad_output.squeeze(1)
            return grad_flat.view(B, C, H, W), None
        
        # 使用显著性掩码进行反向传播
        index_s_list = ctx.index_s_list
        mask_ns_list = ctx.mask_ns_list
        
        grad_reshaped = CrossMergeBase.CrossMerge_saliency(grad_output, index_s_list, mask_ns_list, saliency_mask)
        return grad_reshaped.view(B, C, H, W), None

class CrossMergeBase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys, saliency_mask=None):
        """
        基于显著性掩码的合并
        Args:
            ys: 扫描后的特征图，形状为(B, 1, C, L)
            saliency_mask: 显著性掩码，形状为(B, 1, H, W)
        Returns:
            merged: 合并后的特征图，形状为(B, C, H, W)
        """
        # 保存上下文信息
        B, _, C, L = ys.shape
        H = W = int(math.sqrt(L))  # 从序列长度计算H和W
        ctx.save_for_backward(ys, saliency_mask)
        ctx.shape = (B, C, H, W)

        # 如果没有提供显著性掩码，直接重塑
        if saliency_mask is None:
            return ys.squeeze(1).reshape(B, C, H, W)

        # 计算显著区域和非显著区域
        gt_flat = saliency_mask.view(B, -1)  # [B, H*W]
        salient_mask = gt_flat > 0.5  # [B, H*W]
        non_salient_mask = ~salient_mask  # [B, H*W]

        # 初始化输出
        out = torch.zeros((B, C, H, W), device=ys.device)

        for b in range(B):
            # 计算显著区域和非显著区域的长度
            len_s = salient_mask[b].sum().item()
            len_ns = non_salient_mask[b].sum().item()

            # 获取显著区域和非显著区域的索引
            salient_indices = torch.nonzero(salient_mask[b]).squeeze(-1)
            non_salient_indices = torch.nonzero(non_salient_mask[b]).squeeze(-1)

            # 处理边缘情况
            if len_s == 0:
                salient_indices = torch.tensor([0], device=ys.device)
                len_s = 1
            if len_ns == 0:
                non_salient_indices = torch.tensor([0], device=ys.device)
                len_ns = 1

            # 重排序特征
            y_flat = ys[b, 0]  # [C, L]
            
            # 根据调用类的名称决定使用哪种合并顺序
            caller_name = ctx.__class__.__name__
            
            if caller_name == "CrossMerge_Regular":
                # 先显著区域，后非显著区域
                indices = torch.cat([salient_indices, non_salient_indices])
            elif caller_name == "CrossMerge_Saliency1":
                # 先显著区域（反转），后非显著区域（反转）
                indices = torch.cat([torch.flip(salient_indices, [0]), torch.flip(non_salient_indices, [0])])
            elif caller_name == "CrossMerge_Saliency2":
                # 先非显著区域，后显著区域
                indices = torch.cat([non_salient_indices, salient_indices])
            elif caller_name == "CrossMerge_Saliency3":
                # 先非显著区域（反转），后显著区域（反转）
                indices = torch.cat([torch.flip(non_salient_indices, [0]), torch.flip(salient_indices, [0])])
            else:
                # 默认：先显著区域，后非显著区域
                indices = torch.cat([salient_indices, non_salient_indices])

            # 重排序并重塑
            out[b] = y_flat[:, indices].reshape(C, H, W)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        Args:
            grad_output: 输出梯度，形状为(B, C, H, W)
        Returns:
            grad_ys: 输入梯度，形状为(B, 1, C, L)
            None: 对应saliency_mask的梯度（不需要计算）
        """
        ys, saliency_mask = ctx.saved_tensors
        B, C, H, W = ctx.shape
        L = H * W

        # 如果没有显著性掩码，直接重塑梯度
        if saliency_mask is None:
            return grad_output.reshape(B, C, L).unsqueeze(1), None

        # 计算显著区域和非显著区域
        gt_flat = saliency_mask.view(B, -1)  # [B, H*W]
        salient_mask = gt_flat > 0.5  # [B, H*W]
        non_salient_mask = ~salient_mask  # [B, H*W]

        # 初始化梯度
        grad_ys = torch.zeros_like(ys)

        for b in range(B):
            # 获取显著区域和非显著区域的索引
            salient_indices = torch.nonzero(salient_mask[b]).squeeze(-1)
            non_salient_indices = torch.nonzero(non_salient_mask[b]).squeeze(-1)

            # 处理边缘情况
            if len(salient_indices) == 0:
                salient_indices = torch.tensor([0], device=ys.device)
            if len(non_salient_indices) == 0:
                non_salient_indices = torch.tensor([0], device=ys.device)

            # 根据调用类的名称决定使用哪种合并顺序
            caller_name = ctx.__class__.__name__
            
            if caller_name == "CrossMerge_Regular":
                indices = torch.cat([salient_indices, non_salient_indices])
            elif caller_name == "CrossMerge_Saliency1":
                indices = torch.cat([torch.flip(salient_indices, [0]), torch.flip(non_salient_indices, [0])])
            elif caller_name == "CrossMerge_Saliency2":
                indices = torch.cat([non_salient_indices, salient_indices])
            elif caller_name == "CrossMerge_Saliency3":
                indices = torch.cat([torch.flip(non_salient_indices, [0]), torch.flip(salient_indices, [0])])
            else:
                indices = torch.cat([salient_indices, non_salient_indices])

            # 重排序梯度
            grad_flat = grad_output[b].reshape(C, -1)  # [C, L]
            grad_ys[b, 0] = grad_flat[:, indices]

        return grad_ys, None

# 定义四种扫描模式，全部使用显著性扫描
class CrossScan_Regular(CrossScanBase):
    @staticmethod
    def forward(ctx, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossScanBase.forward(ctx, x, saliency_mask)

class CrossScan_Saliency1(CrossScanBase):
    @staticmethod
    def forward(ctx, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossScanBase.forward(ctx, x, saliency_mask)

class CrossScan_Saliency2(CrossScanBase):
    @staticmethod
    def forward(ctx, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossScanBase.forward(ctx, x, saliency_mask)

class CrossScan_Saliency3(CrossScanBase):
    @staticmethod
    def forward(ctx, x: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossScanBase.forward(ctx, x, saliency_mask)

# 定义四种合并模式，全部使用显著性扫描
class CrossMerge_Regular(CrossMergeBase):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossMergeBase.forward(ctx, ys, saliency_mask)

class CrossMerge_Saliency1(CrossMergeBase):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossMergeBase.forward(ctx, ys, saliency_mask)

class CrossMerge_Saliency2(CrossMergeBase):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossMergeBase.forward(ctx, ys, saliency_mask)

class CrossMerge_Saliency3(CrossMergeBase):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, saliency_mask: Optional[torch.Tensor] = None):
        return CrossMergeBase.forward(ctx, ys, saliency_mask) 