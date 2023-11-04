def calculate_iou(bbox_ref, bbox_tar):
    """calculate the iou of two bbox
    Args:
        bbox_ref (x,y min max): the bbox of reference
        bbox_tar (x,y min max): the bbox of target

    Returns:
        value: the iou value
    """
    x_ref_min, y_ref_min, x_ref_max, y_ref_max = bbox_ref
    x_tar_min, y_tar_min, x_tar_max, y_tar_max = bbox_tar

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_A = max(x_ref_min, x_tar_min)
    y_A = max(y_ref_min, y_tar_min)
    x_B = min(x_ref_max, x_tar_max)
    y_B = min(y_ref_max, y_tar_max)

    # Compute the area of intersection rectangle
    inter_Area = max(0, x_B - x_A + 1) * max(0, y_B - y_A + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box_A_Area = (x_ref_max - x_ref_min + 1) * (y_ref_max - y_ref_min + 1)
    box_B_Area = (x_tar_max - x_tar_min + 1) * (y_tar_max - y_tar_min + 1)

    # NOTICE: different objects may overlap, uses area to determine the object
    if inter_Area / box_A_Area < 0.5:
        return 0

    # calculate the iou
    iou = inter_Area / float(box_A_Area + box_B_Area - inter_Area)
    return iou


box1 = (10, 20, 50, 80)
box2 = (10, 40, 70, 100)
iou = calculate_iou(box2, box2)
print(f"IoU: {iou}")
