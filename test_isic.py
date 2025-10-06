import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from ISIC_loader import *

from model.STMamba import STMamba
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join('')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config    
    # 使用STMamba模型替代UltraLight_VM_UNet
    model = STMamba(
        img_size=config.img_size,
        in_chans=model_cfg['input_channels'],
        num_classes=model_cfg['num_classes'],
        embed_dims=model_cfg.get('embed_dims', [64, 128, 256, 512]),
        depths=model_cfg.get('depths', [2, 2, 9, 2]),
        ssm_d_state=model_cfg.get('ssm_d_state', 16),
        ssm_dt_rank=model_cfg.get('ssm_dt_rank', "auto"),
        ssm_ratio=model_cfg.get('ssm_ratio', 2.0),
        mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
        use_checkpoint=model_cfg.get('use_checkpoint', False),
        use_tree_fusion=model_cfg.get('use_tree_fusion', True),
    )
    
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])


    print('#----------Preparing dataset----------#')
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )



if __name__ == '__main__':
    config = setting_config
    main(config)