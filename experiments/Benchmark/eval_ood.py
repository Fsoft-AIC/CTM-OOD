import os
from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=False
)
# For hydra config path. See configs/paths/default.yaml
os.environ["EXP_DIR"]=str(Path(__file__).parent.resolve()) 

import logging
import data as data
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
import hydra
from core.detection.pipelines.standard import run
from core.detection.methods.utils import CalMeanClass


logger = logging.getLogger(__name__)


def build_forward_fn(cfg, NUM_CLASSES, class_names):
    '''
    Do forward pass and preprocess feature maps. 
    
    ----
    Return:
        forward_fn(batch, return_feature_list, penultimate_feature) 
    '''
    
    # Load model
    net = hydra.utils.instantiate(cfg.net.init_params, num_classes=NUM_CLASSES)
    net.load(cfg.net.ckpt_path)
    logger.info('Model restored! Checkpoint path: %s' % cfg.net.ckpt_path)

    # Set device
    device = "cpu"
    if len(cfg.gpu_ids) > 0:
        device = f"cuda:{cfg.gpu_ids[0]}"
        logger.info(f"Using GPU: {device}")
    net.to(device)

    # Set to eval mode
    net.eval()

    def forward_fn(batch, return_feature_list=False, penultimate_feature=False):
        '''
        Params:
            batch: (X, y)
            return_feature_list: default=False
            penultimate_feature: default=False
        '''
        if penultimate_feature:
            return net.fc(batch.to(device))
        else:
            X = batch[0]
            if return_feature_list: # return feature maps
                logits, features_list = net(X.to(device), return_feature_list=True)                
                for i in range(0, len(features_list)):
                    if features_list[i].ndim == 4:
                        features_list[i] = torch.mean(features_list[i], [2,3])
                return logits, features_list
            else: # return logits only
                logits = net(X.to(device))
                # logits, features_list = net(data.to(device), return_feature_list=True)
                # logits = net.fc(F.normalize(features_list[0]))
                return logits.cpu()

    return forward_fn, net


@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='ood_eval.yaml', version_base="1.2")
def main(cfg):
    # Set seed  
    torch.manual_seed(112)
    torch.cuda.manual_seed(113)
    # np.random.seed(1)
    
    # Log config
    DATA_ROOT = Path(os.environ['TORCH_DATASETS'])
    logger.info(cfg)
    logger.info('Logging experiment to: %s' % cfg.paths.output_dir)
    logger.info(f'Data root: {DATA_ROOT}')

    cudnn.benchmark = True  # boost speed
    if 'data_mean' in cfg.net: # set data mean and std
        data.NORM_PARAMS[cfg.in_dataset]["mean"] = cfg.net.data_mean
        data.NORM_PARAMS[cfg.in_dataset]["std"] = cfg.net.data_std
    logger.info(data.NORM_PARAMS[cfg.in_dataset])

    # Load in-distribution data
    id_data_dict = data.get_id_datasets_dict(DATA_ROOT, cfg.in_dataset)
    NUM_CLASSES = id_data_dict['meta']['num_classes']
    logger.info(f'Using {data.ID2PRINTNAME[cfg.in_dataset]} as typical data')
    
    # Build forward function
    forward_fn, net = build_forward_fn(cfg, NUM_CLASSES, class_names=id_data_dict['meta']["class_names"])

    # Load ood data
    ood_datasets_dict = data.get_ood_datasets_dict(DATA_ROOT, cfg.in_dataset)
    ood_num_examples = len(id_data_dict['ds']['test'])
    # ood_num_examples = len(id_data_dict['ds']['test']) // 5
    # ood_num_examples = None

    # Load detector
    detector = hydra.utils.instantiate(cfg.detector, DATASET_MEAN=data.NORM_PARAMS[cfg.in_dataset]["mean"], DATASET_STD=data.NORM_PARAMS[cfg.in_dataset]["std"])()
    
    # Run pipeline
    run(cfg, id_data_dict, ood_datasets_dict, forward_fn, net, detector, ood_num_examples)

if __name__ == "__main__":
    main()