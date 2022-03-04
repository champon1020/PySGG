import argparse
import torch
import json
import os
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="checkpoint/visualize")
parser.add_argument("--inference_dir", type=str)
parser.add_argument("--frame_id", type=str)
args = parser.parse_args()

args.inference_dir = "checkpoint/vidvrd/motif/sgdet-MotifPredictor/inference/VidVRD_test/"
args.frame_id = "ILSVRC2015_train_00265005.mp4/000010.jpg"

eval_results = torch.load(op.join(args.inference_dir, "eval_results.pytorch"))
visual_info = json.load(open(op.join(args.inference_dir, "visual_info.json")))
vgg_dicts = json.load(open("datasets/vidvrd/VG-SGG-dicts.json"))
idx_to_label = vgg_dicts["idx_to_label"]
idx_to_predicate = vgg_dicts["idx_to_predicate"]

output_dir = op.join(args.output_dir, args.frame_id.split("/")[0])
if not op.exists(output_dir):
    os.mkdir(output_dir)

for i, info in enumerate(tqdm(visual_info)):
    frame_id = "/".join(info["img_file"].split("/")[-2:])
    if frame_id != args.frame_id:
        continue
    print("Start process {}".format(frame_id))
    predictions = eval_results["predictions"][i]

    pred_labels = predictions.get_field("pred_labels") # (N)
    pred_scores = predictions.get_field("pred_scores") # (N)
    rel_pair_idxs = predictions.get_field("rel_pair_idxs") # (M, 2)
    pred_rel_scores = predictions.get_field("pred_rel_scores") # (M, C)
    pred_rel_labels = predictions.get_field("pred_rel_labels") # (M)

    """
    N = pred_labels.shape[0]
    M = pred_rel_labels.shape[0]
    score_map = torch.zeros((N, N, M))

    for i in range(M):
        sbj, obj = rel_pair_idxs[i]
        score_map[sbj, obj, i] = pred_scores[sbj] * pred_scores[obj] * torch.max(pred_rel_scores[i]).item()

    labels = [idx_to_label[str(label.item())] for label in pred_labels]

    for i in range(M):
        pre = pred_rel_labels[i]
        pre = idx_to_predicate[str(pre.item())]
        sns.set(font_scale = 1.2)
        plt.subplots_adjust(top=0.90, left=0.15, bottom=0.20, right=1.0)
        ax = plt.axes()
        ax.title.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.xaxis.labelpad = 10
        h = sns.heatmap(
            score_map[:, :, i],
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            ax=ax,
        )
        h.set_xlabel("object", fontsize=20)
        h.set_ylabel("subject", fontsize=20)
        h.set_title(pre, fontsize=20)
        basename, ext = op.splitext(frame_id)
        frame_number = basename.split("/")[-1]
        output_name = op.join(
            output_dir, f"{frame_number}_att_{i}_{pre}{ext}"
        )
        plt.savefig(output_name)
        plt.close()
    """

    n_rel_classes = pred_rel_scores.shape[1]
    N = pred_labels.shape[0]
    M = pred_rel_labels.shape[0]
    score_map = torch.zeros((N, N, n_rel_classes))

    for i in range(M):
        sbj, obj = rel_pair_idxs[i]
        for j in range(n_rel_classes):
            score_map[sbj, obj, j] = pred_scores[sbj] * pred_scores[obj] * pred_rel_scores[i, j].item()

    sorted_ind = pred_labels.argsort()
    score_map = score_map[sorted_ind][:, sorted_ind]
    pred_labels = pred_labels[sorted_ind]

    labels = [idx_to_label[str(label.item())] for label in pred_labels]

    for j in range(1, n_rel_classes):
        pre = idx_to_predicate[str(j)]
        sns.set(font_scale = 2.7)
        plt.subplots_adjust(top=0.90, left=0.13, bottom=0.20, right=0.90)
        ax = plt.axes()
        ax.title.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.xaxis.labelpad = 10
        sns.set(rc={"figure.figsize": (9.0, 6.0)})
        h = sns.heatmap(
            score_map[:, :, j],
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            annot_kws={"fontsize": 30},
            ax=ax,
        )
        h.set_xlabel("object", fontsize=30)
        h.set_ylabel("subject", fontsize=30)
        h.set_title(pre, fontsize=30)
        h.collections[0].colorbar.ax.tick_params(labelsize=30)
        basename, ext = op.splitext(frame_id)
        frame_number = basename.split("/")[-1]
        output_name = op.join(
            output_dir, f"{frame_number}_att_{pre}{ext}"
        )
        plt.savefig(output_name)
        plt.close()
