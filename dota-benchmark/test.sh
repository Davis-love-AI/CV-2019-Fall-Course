export NGPUS=4
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=50012 tools/test_net.py --config-file configs/e2e_faster_rcnn_dota_R_50_FPN_1x.yaml
