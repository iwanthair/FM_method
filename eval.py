
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

TITLE = 'Palette SE PE Test'
RESULT_PATH = 'results_test/20250813-172223/test/'


def predict_ssim(gt_img, pred_img):
    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img)

    score = ssim(gt_np, pred_np, data_range=255)
    return score

def predict_iou(gt_img, pred_img, threshold=128):
    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img)

    gt_bin = (gt_np >= threshold).astype(np.uint8)
    pred_bin = (pred_np >= threshold).astype(np.uint8)

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()

    iou = intersection / union if union > 0 else 1.0
    return iou


def predict_accuracy(gt1, pred2):
    gt1 = gt1.point(lambda x: 0 if x < 128 else 255, '1')
    pred2 = pred2.point(lambda x: 0 if x < 128 else 255, '1')
    # occupied_accuracy and free_accuracy
    occupied = 0
    free = 0
    gt_occupied = 0
    gt_free = 0
    for i in range(gt1.size[0]):
        for j in range(gt1.size[1]):
            if gt1.getpixel((i, j)) == 255:
                gt_occupied += 1
                if pred2.getpixel((i, j)) == 255:
                    occupied += 1
            else:
                gt_free += 1
                if pred2.getpixel((i, j)) == 0:
                    free += 1
    occupied_accuracy = occupied / gt_occupied if gt_occupied > 0 else 0
    free_accuracy = free / gt_free if gt_free > 0 else 0
    return occupied_accuracy, free_accuracy


def get_img(gt_path, pred_path):
    gt_name = gt_path.split('/')[-1].split('.')[0]
    pred_name = pred_path.split('/')[-1].split('.')[0]
    if gt_name != pred_name:
        raise ValueError(f'Ground truth image {gt_name} does not match predicted image {pred_name}')
    gt_img = Image.open(gt_path).convert('L')  # convert to grayscale
    pred_img = Image.open(pred_path).convert('L')  # convert to grayscale
    return gt_img, pred_img



if __name__ == '__main__':
    dir_list = os.listdir(RESULT_PATH)
    print('dir_list:', dir_list)

    gt_paths = []
    pred_paths = []
    for dir in dir_list:
        if dir.startswith('gt'):
            gt_paths = [os.path.join(RESULT_PATH, dir, i) for i in os.listdir(os.path.join(RESULT_PATH, dir)) if i.endswith('.png')]
        elif dir.startswith('pred'):
            pred_paths = [os.path.join(RESULT_PATH, dir, i) for i in os.listdir(os.path.join(RESULT_PATH, dir)) if i.endswith('.png')]
        else:
            raise ValueError(f'Unknown directory {dir} in {RESULT_PATH}')

    gt_paths = [path.replace('\\', '/') for path in gt_paths]
    pred_paths = [path.replace('\\', '/') for path in pred_paths]
    print(f'Number of ground truth images: {len(gt_paths)}, Number of predicted images: {len(pred_paths)}')

    if len(gt_paths) != len(pred_paths):
        raise ValueError(f'Number of ground truth images {len(gt_paths)} does not match number of predicted images {len(pred_paths)}')
    else:
        print(f'Number of ground truth images: {len(gt_paths)}, Number of predicted images: {len(pred_paths)}')

    # average accuracy
    total_occupied_accuracy = 0
    total_free_accuracy = 0
    total_ssim = 0
    total_iou = 0
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt_img, pred_img = get_img(gt_path, pred_path)
        
        occupied_accuracy, free_accuracy = predict_accuracy(gt_img, pred_img)
        ssim_score = predict_ssim(gt_img, pred_img)
        iou_score = predict_iou(gt_img, pred_img)
        total_occupied_accuracy += occupied_accuracy
        total_free_accuracy += free_accuracy
        total_ssim += ssim_score
        total_iou += iou_score

    avg_occupied_accuracy = total_occupied_accuracy / len(gt_paths)
    avg_free_accuracy = total_free_accuracy / len(gt_paths)
    avg_ssim = total_ssim / len(gt_paths)
    avg_iou = total_iou / len(gt_paths)

    print(f'Average occupied accuracy for {TITLE}: {avg_occupied_accuracy:.4f}, Average free accuracy: {avg_free_accuracy:.4f}, Average SSIM: {avg_ssim:.4f}, Average IoU: {avg_iou:.4f}')