import logging
import pickle
from core.detection.evaluation import Evaluator
import torchvision.transforms as trn
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from pathlib import Path

logger = logging.getLogger(__name__)

def run(cfg, id_data_dict, ood_datasets_dict, forward_fn, net, detector, ood_num_examples):
    NUM_CLASSES = id_data_dict['meta']['num_classes']    

    # /////////////// Detection Prelims ///////////////
    ref_dataloader = torch.utils.data.DataLoader(id_data_dict['ds']['train'],
                                                 batch_size=cfg.test_bs, shuffle=False,
                                                 num_workers=cfg.prefetch, pin_memory=False)
    detector.adapt(forward_fn, ref_dataloader, num_classes=NUM_CLASSES, net=net)

    # /////////////// OOD Detection ///////////////
    evaluator = Evaluator(ood_num_examples, cfg.test_bs, cfg.num_to_avg)
    logger.info("Using clean testset as id data")
    
    id_acc = evaluator.eval_in_acc(
        forward_fn, f"{cfg.in_dataset}", id_data_dict['ds']['test'])
    logger.info(f"In-Dist Acc [{cfg.in_dataset}]: {id_acc*100:0.4f}%")
    evaluator.compute_in_score(
        forward_fn, detector, f"{cfg.in_dataset}", id_data_dict['ds']['test'])
    
    for ood_name, ood_dataset in ood_datasets_dict["ds"]['far'].items():
        logger.info(f'{ood_name} Detection')
        evaluator.eval_ood(forward_fn, detector, id_name=f"{cfg.in_dataset}",
                           ood_name=f"{ood_name}", ood_data=ood_dataset, is_near=False)
    
    for ood_name, ood_dataset in ood_datasets_dict["ds"]['near'].items():
        logger.info(f'{ood_name} Detection')
        evaluator.eval_ood(forward_fn, detector, id_name=f"{cfg.in_dataset}",
                           ood_name=f"{ood_name}", ood_data=ood_dataset, is_near=True)

    # /////////////// Mean Results ///////////////
    logger.info(f"Test Results!!!!!\n{evaluator.df}")
    far_df = evaluator.df[evaluator.df["is_near"] == False]
    near_df = evaluator.df[evaluator.df["is_near"]]
    logger.info(f"Mean Far-OOD Test Results\n{far_df[['auroc', 'aupr', 'fpr']].mean()}")
    logger.info(f"Mean Near-OOD Test Results\n{near_df[['auroc', 'aupr', 'fpr']].mean()}")

    # evaluator.reset()
    if cfg.save2csv:
        save_path = Path(cfg.paths.output_dir) / (str(detector) + ".csv")
        # if save_path.is_file():
        #     evaluator.df.to_csv(save_path, mode='a', header=False, index=False)
        # else:
        score_path = Path(cfg.paths.output_dir) / (str(detector) + ".pth")
        torch.save({"id_scores": evaluator.in_score_dict, "ood_scores": evaluator.ood_score_dict}, score_path)
        evaluator.df.to_csv(save_path, index=False)
