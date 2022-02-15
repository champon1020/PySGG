# detector pre-training
.PHONY: train_detector_ag
train_detector_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/detector_pretrain_net.py \
		--config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" \
		SOLVER.MAX_ITER 40000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.RELATION_ON False \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/pretrained_faster_rcnn/ag

.PHONY: train_detector_vidvrd
train_detector_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/detector_pretrain_net.py \
		--config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" \
		SOLVER.MAX_ITER 40000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.RELATION_ON False \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/pretrained_faster_rcnn/vidvrd

# training on AG
.PHONY: train_motif_ag
train_motif_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "MotifPredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/ag/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/motif/

.PHONY: train_gpsnet_ag
train_gpsnet_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "GPSNetPredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/ag/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/gpsnet/

.PHONY: train_vctree_ag
train_vctree_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 4 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "VCTreePredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/ag/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/vctree/

# training on VidVRD
.PHONY: train_motif_vidvrd
train_motif_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "MotifPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/vidvrd/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/motif/

.PHONY: train_gpsnet_vidvrd
train_gpsnet_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "GPSNetPredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/vidvrd/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/gpsnet/

.PHONY: train_vctree_vidvrd
train_vctree_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 8 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "VCTreePredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/vidvrd/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/vctree/

.PHONY: train_video_transformer_vidvrd
train_video_transformer_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_train_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		--skip-test \
		SOLVER.MAX_ITER 20000 \
		SOLVER.STEPS "(30000, 45000)" \
		SOLVER.VAL_PERIOD 10000 \
		SOLVER.CHECKPOINT_PERIOD 10000 \
		SOLVER.PRE_VAL False \
		SOLVER.IMS_PER_BATCH 4 \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "TransformerPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoint/pretrained_faster_rcnn/vidvrd/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/transformer/ \
		VIDEO_INPUT True


# test on AG
.PHONY: test_motif_ag
test_motif_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "MotifPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/ag/motif/sgdet-MotifPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/motif/sgdet-MotifPredictor

	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		TEST.IMS_PER_BATCH 4 \
		TEST.MODE "phrdet" \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "MotifPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/ag/motif/sgdet-MotifPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/motif/sgdet-MotifPredictor


.PHONY: test_gpsnet_ag
test_gpsnet_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "GPSNetPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/ag/gpsnet/sgdet-GPSNetPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/gpsnet/sgdet-GPSNetPredictor

	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		TEST.IMS_PER_BATCH 4 \
		TEST.MODE "phrdet" \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "GPSNetPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/ag/gpsnet/sgdet-GPSNetPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/gpsnet/sgdet-GPSNetPredictor


.PHONY: test_vctree_ag
test_vctree_ag:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "VCTreePredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/ag/vctree/sgdet-VCTreePredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/vctree/sgdet-VCTreePredictor

	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "ag" \
		TEST.IMS_PER_BATCH 4 \
		TEST.MODE "phrdet" \
		DATASETS.TRAIN "('AG_train',)" \
		DATASETS.VAL "('AG_test',)" \
		DATASETS.TEST "('AG_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 37 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 27 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "VCTreePredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/ag/vctree/sgdet-VCTreePredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/ag/vctree/sgdet-VCTreePredictor


# test on VidVRD
.PHONY: test_motif_vidvrd
test_motif_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "MotifPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/vidvrd/motif/sgdet-MotifPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/motif/sgdet-MotifPredictor

	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		TEST.IMS_PER_BATCH 4 \
		TEST.MODE "phrdet" \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "MotifPredictor" \
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/vidvrd/motif/sgdet-MotifPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/motif/sgdet-MotifPredictor


.PHONY: test_gpsnet_vidvrd
test_gpsnet_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "GPSNetPredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/vidvrd/gpsnet/sgdet-GPSNetPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/gpsnet/sgdet-GPSNetPredictor

	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		TEST.IMS_PER_BATCH 4 \
		TEST.MODE "phrdet" \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "GPSNetPredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/vidvrd/gpsnet/sgdet-GPSNetPredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/gpsnet/sgdet-GPSNetPredictor


.PHONY: test_vctree_vidvrd
test_vctree_vidvrd:
	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		TEST.IMS_PER_BATCH 4 \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "VCTreePredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/vidvrd/vctree/sgdet-VCTreePredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/vctree/sgdet-VCTreePredictor

	python -m torch.distributed.launch --master_port 10028 --nproc_per_node=4 tools/relation_test_net.py \
		--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
		--dataset "vidvrd" \
		TEST.IMS_PER_BATCH 4 \
		TEST.MODE "phrdet" \
		DATASETS.TRAIN "('VidVRD_train',)" \
		DATASETS.VAL "('VidVRD_test',)" \
		DATASETS.TEST "('VidVRD_test',)" \
		MODEL.ROI_BOX_HEAD.NUM_CLASSES 36 \
		MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES 1 \
		MODEL.ROI_RELATION_HEAD.NUM_CLASSES 133 \
		MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
		MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
		MODEL.ROI_RELATION_HEAD.PREDICTOR "VCTreePredictor"\
		MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS True \
		MODEL.WEIGHT ./checkpoint/vidvrd/vctree/sgdet-VCTreePredictor/model_final.pth \
		GLOVE_DIR datasets/glove \
		OUTPUT_DIR ./checkpoint/vidvrd/vctree/sgdet-VCTreePredictor
