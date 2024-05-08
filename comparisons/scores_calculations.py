import random
# Helper function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate intersection area
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = max(0, x_inter2 - x_inter1 + 1)
    height_inter = max(0, y_inter2 - y_inter1 + 1)
    area_inter = width_inter * height_inter

    # Calculate union area
    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    area_union = area1 + area2 - area_inter

    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0

    return iou
# Define some dummy data
num_faces = 10
num_detections = 12

# Generate dummy ground truth bounding boxes
ground_truth_boxes = [(random.randint(10, 500), random.randint(10, 500), random.randint(50, 600), random.randint(50, 600)) for _ in range(num_faces)]

# Generate dummy predicted bounding boxes
predicted_boxes = [(random.randint(10, 500), random.randint(10, 500), random.randint(50, 600), random.randint(50, 600)) for _ in range(num_detections)]

# Calculate IoU between ground truth and predicted boxes
ious = []
for gt_box in ground_truth_boxes:
    max_iou = 0
    for pred_box in predicted_boxes:
        iou = calculate_iou(gt_box, pred_box)
        if iou > max_iou:
            max_iou = iou
    ious.append(max_iou)

# Calculate precision, recall, and F1-score
true_positives = sum(iou > 0.5 for iou in ious)
false_positives = num_detections - true_positives
false_negatives = num_faces - true_positives

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print the dummy table
print("Metric\tValue")
print("-------\t-----")
print(f"Precision\t{precision:.2f}")
print(f"Recall\t{recall:.2f}")
print(f"F1-score\t{f1_score:.2f}")
print(f"Average IoU\t{sum(ious) / len(ious):.2f}")

