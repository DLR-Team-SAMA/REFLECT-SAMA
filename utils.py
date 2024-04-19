def bboxes_to_rois(image, bboxes):
    rois = []
    for bbox in bboxes:
        roi = image.crop(bbox)
        rois.append(roi)
    return(rois)
