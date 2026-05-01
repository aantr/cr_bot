import cv2
import numpy as np

def non_max_suppression(rectangles, threshold=0.8):
    """
    Apply non-maximum suppression to remove overlapping rectangles
    """
    if len(rectangles) == 0:
        return []
    
    # Convert to integer coordinates
    boxes = np.array(rectangles, dtype=np.float32)
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Calculate area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by confidence (assuming 4th element is confidence)
    if boxes.shape[1] == 5:
        idxs = np.argsort(boxes[:, 4])
    else:
        idxs = np.argsort(y2 - y1)  # Fallback: sort by height
    
    # Initialize list for picked rectangles
    picked = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        picked.append(i)
        
        # Find overlapping area
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete indexes with overlap > threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
    
    return [rectangles[i] for i in picked]

def find_all_matches_nms(image, template, threshold=0.8, nms_threshold=0.5, method=cv2.TM_CCOEFF_NORMED):
    """
    Find all matches with NMS to eliminate duplicates
    """
    h, w = template.shape[:2]
    result = cv2.matchTemplate(image, template, method)
    
    # Find all matches above threshold
    locations = np.where(result >= threshold)
    
    # Create rectangles with confidence scores
    rectangles = []
    for pt in zip(*locations[::-1]):
        confidence = result[pt[1], pt[0]]
        rectangles.append([pt[0], pt[1], w, h, confidence])
    
    # Apply non-maximum suppression
    rectangles = non_max_suppression(rectangles, nms_threshold)
    
    return rectangles


def find_and_draw_pattern(big_img, pattern, threshold=0.8):
    """
    Find pattern in big image and draw rectangle around the best match
    
    Args:
        big_image_path: Path to the large image
        pattern_path: Path to the pattern/template image
        threshold: Matching threshold (0-1), higher = more strict
    """
    
    if big_img is None or pattern is None:
        print("Error: Could not load images")
        return None
    
    # Get pattern dimensions
    pattern_h, pattern_w = pattern.shape[:2]

    matches = find_all_matches_nms(big_img, pattern, threshold=threshold, nms_threshold=0.3)
    # Draw rectangles for all matches
    for (x, y, w, h, confidence) in matches:
        # Get top-left corner of match
        top_left = (x, y)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Draw rectangle
        cv2.rectangle(big_img, top_left, bottom_right, (0, 255, 0), 2)
        
        # Optional: Add text with confidence score
        cv2.putText(big_img, f"Match: {confidence:.2f}", 
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return big_img, matches