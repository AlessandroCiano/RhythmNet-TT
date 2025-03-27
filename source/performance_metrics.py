import torch
import warnings

########## classification metrics ##########
def get_TP_TN_FP_FN(predicted_indices, target_indices, num_classes,device):
    #torch.set_default_device(device) #Send all tensors to the correct device
    # Convert indices to one-hot encoding
    predicted_onehot = torch.eye(num_classes,device=device)[predicted_indices.type(torch.int)]
    target_onehot = torch.eye(num_classes,device=device)[target_indices.type(torch.int)]
    predicted = predicted_onehot.reshape(-1, num_classes)
    target = target_onehot.reshape(-1, num_classes)
    
    TP, TN, FP, FN = torch.zeros(num_classes,device=device), torch.zeros(num_classes,device=device), torch.zeros(num_classes,device=device), torch.zeros(num_classes,device=device)

    for i in range(num_classes):
        TP[i] = torch.sum((predicted[:, i] == 1) & (target[:, i] == 1))
        TN[i] = torch.sum((predicted[:, i] == 0) & (target[:, i] == 0))
        FP[i] = torch.sum((predicted[:, i] == 1) & (target[:, i] == 0))
        FN[i] = torch.sum((predicted[:, i] == 0) & (target[:, i] == 1))
    return TP, TN, FP, FN

def get_metrics(TP, TN, FP, FN):
    # Calculate the metrics for each class and overall (Macro-averaging and Micro-averaging)
    accuracy_single = (TP + TN) / (TP + TN + FP + FN)
    accuracy_single = torch.nan_to_num(accuracy_single, nan=0.0)
    accuracy_micro_avg = (torch.sum(TP) + torch.sum(TN)) / (torch.sum(TP) + torch.sum(TN) + torch.sum(FP) + torch.sum(FN))
    accuracy_macro_avg = torch.mean(accuracy_single)
    accuracy = {'single_accuracy': accuracy_single.cpu(), 'micro_avg_accuracy': accuracy_micro_avg.cpu(), 'macro_avg_accuracy': accuracy_macro_avg.cpu()}
    
    # Calculate precision with division handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision_single = TP / (TP + FP)
    precision_single = torch.nan_to_num(precision_single, nan=0.0)
    precision_macro_avg = torch.mean(precision_single)
    precision = {'single_precision': precision_single.cpu(), 'macro_avg_precision': precision_macro_avg.cpu()}
    
    # Calculate recall with division handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recall_single = TP / (TP + FN)
    recall_single = torch.nan_to_num(recall_single, nan=0.0)
    recall_macro_avg = torch.mean(recall_single)
    recall = {'single_recall': recall_single.cpu(), 'macro_avg_recall': recall_macro_avg.cpu()}
    
    # Calculate F1 score with division handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_single = (2 * precision_single * recall_single) / (precision_single + recall_single)
    f1_single = torch.torch.nan_to_num(f1_single, nan=0.0)
    f1_macro_avg = torch.mean(f1_single)
    f1 = {'single_f1': f1_single.cpu(), 'macro_avg_f1': f1_macro_avg.cpu()}
    
    return accuracy, precision, recall, f1



########## bounding box metrics ##########
def get_iou_1d(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2

    # Compute intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter_length = max(0 , inter_end - inter_start)  # If negative, no intersection

    # Compute union
    total_length = (end1 - start1) + (end2 - start2) - inter_length

    return inter_length / total_length if total_length > 0 else 0


def get_iou_1d_batch(target_intervals, predicted_intervals):

    # Calculate starts and ends of intervals
    start1, end1 = target_intervals[:, 0], target_intervals[:, 1]
    start2, end2 = predicted_intervals[:, 0], predicted_intervals[:, 1]

    # Compute intersection using broadcasting
    inter_start = torch.max(start1.unsqueeze(1), start2.unsqueeze(0))  # Broadcasting over both sets of intervals
    inter_end = torch.min(end1.unsqueeze(1), end2.unsqueeze(0))

    # Calculate intersection length (must be non-negative)
    inter_length = torch.max(inter_end - inter_start, torch.zeros_like(inter_end))

    # Compute union
    total_length = (end1.unsqueeze(1) - start1.unsqueeze(1)) + (end2.unsqueeze(0) - start2.unsqueeze(0)) - inter_length

    # Compute IoU (avoid division by zero)
    iou_matrix = torch.where(total_length > 0, inter_length / total_length, torch.zeros_like(inter_length))

    return iou_matrix


def get_windows_TP_FP_FN(predicted_intervals, target_intervals, predicted_classes, real_classes,device ,iou_threshold=0.5):
    
    iou_threshold=torch.tensor(iou_threshold).to(device) #Send the threshold to the same device
    predicted_intervals = predicted_intervals.reshape(-1, 2)
    target_intervals = target_intervals.reshape(-1, 2)
    predicted_classes = predicted_classes.reshape(-1)
    real_classes = real_classes.reshape(-1)

    # Filter out predicted intervals with class 4 (no-obj)
    predicted_mask = predicted_classes != 4
    predicted_intervals = predicted_intervals[predicted_mask]
    predicted_classes = predicted_classes[predicted_mask]

    # Filter out target intervals with class 4 (not annotated)
    target_mask = real_classes != 4
    target_intervals = target_intervals[target_mask]
    real_classes = real_classes[target_mask]

    TP, FP, FN = 0, 0, 0

    # Handle edge cases
    if len(predicted_intervals) == 0:
        FN = len(target_intervals)
        return TP, FP, FN
    if len(target_intervals) == 0:
        FP = len(predicted_intervals)
        return TP, FP, FN


    #Compute IoU matrix
    iou_matrix = get_iou_1d_batch(target_intervals,predicted_intervals)
    """# Compute IoU matrix
    iou_matrix = torch.zeros((len(target_intervals), len(predicted_intervals)),device=device)
    for i, target_interval in enumerate(target_intervals):
        for j, predicted_interval in enumerate(predicted_intervals):
            iou_matrix[i, j] = get_iou_1d(target_interval, predicted_interval)"""

    # Calculate TP: number of targets with at least one prediction >= iou_threshold
    TP = torch.sum(torch.max(iou_matrix, axis=1)[0] >= iou_threshold)

    # Calculate FP: predictions with all IoU < threshold
    FP = torch.sum(torch.max(iou_matrix, axis=0)[0] < iou_threshold)

    # Calculate FN: targets not matched by any prediction
    FN = len(target_intervals) - TP

    return TP, FP, FN

def get_windows_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
    return precision if type(precision)==float else precision.cpu(), recall if type(recall)==float else recall.cpu(), f1 if type(f1)==float else f1.cpu()

