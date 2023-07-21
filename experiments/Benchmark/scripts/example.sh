# Evaluate all method 
python eval_ood.py -m run_name=cifar10_wrn in_dataset=cifar10 net=wrn detector='glob(*,exclude=base)' test_bs=400 save2csv=T
python eval_ood.py -m run_name=cifar10_densenet in_dataset=cifar10 net=densenet detector='glob(*,exclude=base)' test_bs=128 save2csv=T
python eval_ood.py -m run_name=cifar100_wrn in_dataset=cifar100 net=wrn detector='glob(*,exclude=base)' test_bs=400 save2csv=T
python eval_ood.py -m run_name=cifar100_densenet in_dataset=cifar100 net=densenet detector='glob(*,exclude=base)' test_bs=128 save2csv=T

# Evaluate single methodm, omit save2csv to not save csv file
# Available detectors, see configs/detector/, exclude base detector
python eval_ood.py in_dataset=cifar10 net=wrn run_name=cifar10_wrn detector=mahalanobis test_bs=400 save2csv=T
python eval_ood.py in_dataset=cifar10 net=wrn run_name=cifar10_wrn detector=angle_maha detector.add_bg_score=False test_bs=400 save2csv=T
python eval_ood.py in_dataset=imagenet net=resnet50 run_name=imagnenet_resnet50 detector=osa detector.layer_index=0 detector.add_bg_score=False detector.remove_mean_dir=True detector.class_dir=mean detector.weight_type=inv detector.rank_rtol=null test_bs=512 save2csv=T num_to_avg=5

# Run with checkpoint path
python eval_ood.py in_dataset=cifar10 net=wrn net.ckpt_path=<path> run_name=cifar10_wrn detector=mahalanobis test_bs=400 save2csv=T