import torch
from torch.utils.data import DataLoader
import timm
from dataset_isic import NPY_datasets
from tensorboardX import SummaryWriter
from model import STMamba

from engine import *
import os
import sys
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import dilation, disk
import matplotlib.pyplot as plt

from utils_isic import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def get_gt_bnd(gt):
    """
    获取边界
    
    Args:
        gt: 输入掩码，可以是2D或3D数组
    
    Returns:
        边界掩码
    """
    # 确保输入是3D数组 [batch, height, width]
    if len(gt.shape) == 2:
        gt = np.expand_dims(gt, axis=0)
    
    # 阈值处理
    gt = (gt > 50).astype(np.uint8).copy()
    bnd = np.zeros_like(gt).astype(np.uint8)
    
    for i in range(gt.shape[0]):
        _mask = gt[i]
        # 处理二值图像
        _gt = _mask.astype(np.uint8).copy()
        if _gt.max() > 0:  # 确保有前景
            _gt_dil = dilation(_gt, disk(5))
            bnd[i][_gt_dil - _gt == 1] = 1
    
    return bnd


def overlay_boundaries(img, pred_mask, gt_mask, output_path):
    """
    Overlay segmentation boundaries on original image
    
    Args:
        img: Original image as numpy array
        pred_mask: Prediction mask as numpy array
        gt_mask: Ground truth mask as numpy array
        output_path: Path to save the overlay image
    """
    # Resize image if needed
    if img.shape[:2] != pred_mask.shape:
        img = cv2.resize(img, (pred_mask.shape[1], pred_mask.shape[0]))
    
    # Convert to float32 for processing
    rgb_img = np.float32(img) / 255
    
    # Get prediction boundary (绿色)
    pred_boundary = get_gt_bnd(pred_mask).squeeze(0)
    
    # Get ground truth boundary (红色)
    gt_boundary = get_gt_bnd(gt_mask).squeeze(0)
    
    # Create RGB mask for boundaries
    h, w = pred_boundary.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
    
    # 设置预测边界为绿色
    rgb_mask[pred_boundary == 1] = [0, 1, 0]  # Green for prediction
    
    # 设置真实标签边界为红色
    rgb_mask[gt_boundary == 1] = [1, 0, 0]  # Red for ground truth
    
    # Overlay function
    def show_cam_on_image(img, mask):
        replace_mask = np.any(mask > 0, axis=2)
        img[replace_mask] = mask[replace_mask]
        return np.uint8(255 * img)
    
    # Create overlay image
    overlay_img = show_cam_on_image(rgb_img, rgb_mask)
    
    # Save the overlay image
    plt.imsave(output_path, overlay_img)


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=False)

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'STMamba':
        model = STMamba(
            encoder_name="tiny_0230s",
            in_channels=3,
            num_classes=config.num_classes,
        )
        # Load the specified model weights
        model_weight_path = "/root/autodl-tmp/msvm-unet/results/ISIC2017_Best/checkpoints/best-epoch106-loss0.2314.pth"
        print(f'Loading model weights from: {model_weight_path}')
        model_weight = torch.load(model_weight_path)
        model.load_state_dict(model_weight, strict=False)
    else:
        raise Exception('network is not right!')
    model = model.cuda()
    model.eval()

    cal_params_flops(model, 256, logger)

    print('#----------Exporting segmentation results----------#')
    # Create directories for segmentation results
    seg_output_dir = os.path.join(outputs, 'ISIC17_segmentation_results')
    boundary_output_dir = os.path.join(outputs, 'ISIC17_boundary_results')
    overlay_output_dir = os.path.join(outputs, 'ISIC17_overlay_results')
    
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    if not os.path.exists(boundary_output_dir):
        os.makedirs(boundary_output_dir)
    if not os.path.exists(overlay_output_dir):
        os.makedirs(overlay_output_dir)
    
    with torch.no_grad():
        for i, (img, mask, img_path) in enumerate(val_loader):
            # 确保输入数据是float类型
            img = img.float().cuda()
            output = model(img)
            
            # Convert output to binary mask
            pred = torch.sigmoid(output).cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            
            # 转换真实标签为二值图像
            gt_mask = mask.numpy()
            gt_binary = (gt_mask > 0.5).astype(np.uint8) * 255
            
            # Get filename from path
            filename = os.path.basename(img_path[0]).split('.')[0]
            
            # Save segmentation result
            seg_path = os.path.join(seg_output_dir, filename + '_pred.png')
            pred_img = Image.fromarray(pred_binary[0, 0]).convert('L')
            pred_img.save(seg_path)
            
            try:
                # Get boundary from prediction
                boundary_path = os.path.join(boundary_output_dir, filename + '_boundary.png')
                pred_boundary = get_gt_bnd(pred_binary[0, 0])
                cv2.imwrite(boundary_path, pred_boundary[0] * 255)
                
                # Save overlay result with both boundaries
                original_img = np.array(Image.open(img_path[0]).convert('RGB'))
                overlay_path = os.path.join(overlay_output_dir, filename + '_overlay.png')
                overlay_boundaries(original_img, pred_binary[0, 0], gt_binary[0, 0], overlay_path)
                
                print(f'Processed {i+1}/{len(val_loader)}: {filename}')
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    print(f'Segmentation results saved to: {seg_output_dir}')
    print(f'Boundary results saved to: {boundary_output_dir}')
    print(f'Overlay results saved to: {overlay_output_dir}')


if __name__ == '__main__':
    config = setting_config
    main(config)