#export NGPUS=4
export NGPUS=1

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=50012 tools/train_net.py --config-file configs/e2e_faster_rcnn_dota_R_50_FPN_1x.yaml

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=50012 tools/train_net.py --config-file configs/e2e_faster_rcnn_dota_R_50_FPN_1x.yaml

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=50012 tools/train_net.py --config-file configs/e2e_faster_rcnn_dota_scrdet_R_50_FPN_1x.yaml

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=50012 tools/train_net.py --config-file configs/e2e_faster_rcnn_dota_scrdet_R_50_FPN_1x.yaml

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=50012 tools/train_net.py --config-file configs/retinanet_dota_R-101-FPN_1x.yaml
