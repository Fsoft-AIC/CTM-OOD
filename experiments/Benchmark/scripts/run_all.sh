python eval_ood.py -m run_name=custom_aug/cifar10_wrn in_dataset=cifar10 net=wrn detector='glob(*,exclude=base)' test_bs=128 save2csv=T
# 
python eval_ood.py -m run_name=custom_aug/cifar10_densenet in_dataset=cifar10 net=densenet detector='glob(*,exclude=base)' test_bs=128 save2csv=T
# 
python eval_ood.py -m run_name=custom_aug/cifar100_wrn in_dataset=cifar100 net=wrn detector='glob(*,exclude=base)' test_bs=128 save2csv=T
# 
python eval_ood.py -m run_name=custom_aug/cifar100_densenet in_dataset=cifar100 net=densenet detector='glob(*,exclude=[base])' test_bs=128 save2csv=T
#