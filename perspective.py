import cv2
import numpy as np


def _is_reasonable_pixel_per_cm(pixel_per_cm: float, min_ppc: float = 15.0, max_ppc: float = 200.0) -> bool:
    if pixel_per_cm is None:
        return False
    try:
        v = float(pixel_per_cm)
    except Exception:
        return False
    if not np.isfinite(v):
        return False
    if v < min_ppc or v > max_ppc:
        return False
    return True


def estimate_plate_pixel_per_cm(image, plate_diameter_cm: float = 25.0):
    if image is None:
        return None
    if plate_diameter_cm is None or not np.isfinite(float(plate_diameter_cm)) or float(plate_diameter_cm) <= 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape[:2]
    min_r = int(max(30, min(h, w) * 0.18))
    max_r = int(max(min_r + 1, min(h, w) * 0.48))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.25,
        param1=120,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None
    circles = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    if circles.size == 0:
        return None
    
    # Get the largest circle
    best_c = max(circles, key=lambda x: x[2])
    r = float(best_c[2])
    
    if not np.isfinite(r) or r <= 1:
        return None
    return float((2.0 * r) / float(plate_diameter_cm))

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect

def _merge_close_positions(positions, min_gap):
    if len(positions) == 0:
        return np.array([], dtype=np.float32)
    positions = np.sort(np.array(positions, dtype=np.float32))
    merged = [positions[0]]
    for p in positions[1:]:
        if abs(p - merged[-1]) >= min_gap:
            merged.append(p)
    return np.array(merged, dtype=np.float32)

def _estimate_spacing_from_positions(positions, min_diff=6.0, max_diff=240.0):
    if positions is None or len(positions) < 6:
        return None
    pos = _merge_close_positions(positions, min_gap=3.0)
    if len(pos) < 6:
        return None
    diffs = np.diff(pos)
    diffs = diffs[(diffs >= min_diff) & (diffs <= max_diff)]
    if len(diffs) < 6:
        return None
    med = float(np.median(diffs))
    inlier = diffs[(diffs >= 0.6 * med) & (diffs <= 1.6 * med)]
    if len(inlier) >= 6:
        return float(np.median(inlier))
    return med

def estimate_grid_pixel_per_cm(image):
    if image is None:
        return 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape[:2]
    klen = int(max(20, min(60, min(h, w) // 20)))

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
    hlines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vlines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v, iterations=1)
    grid = cv2.bitwise_or(hlines, vlines)

    edges = cv2.Canny(grid, 50, 150)
    min_len = int(max(30, min(h, w) * 0.08))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=min_len, maxLineGap=12)
    if lines is None:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120, minLineLength=40, maxLineGap=10)
        if lines is None:
            return 1.0

    horiz_pos = []
    vert_pos = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang > 180:
            ang = ang % 180
        if ang < 15 or ang > 165:
            horiz_pos.append((y1 + y2) * 0.5)
        elif abs(ang - 90) < 15:
            vert_pos.append((x1 + x2) * 0.5)

    sy = _estimate_spacing_from_positions(horiz_pos)
    sx = _estimate_spacing_from_positions(vert_pos)

    if sx is None and sy is None:
        return 1.0
    if sx is None:
        return float(sy)
    if sy is None:
        return float(sx)
    return float((sx + sy) * 0.5)

def global_perspective_rectify(image, marker_pts, grid_size_cm):
    rect = order_points(marker_pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Use the max dimension as the side of the square
    side = max(maxWidth, maxHeight)
    
    # Center of the marker in the original image
    center = np.mean(rect, axis=0)
    
    # Define the destination square centered at 'center'
    half = side / 2.0
    dst = np.array([
        [center[0] - half, center[1] - half],
        [center[0] + half, center[1] - half],
        [center[0] + half, center[1] + half],
        [center[0] - half, center[1] + half]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, M, (w, h))
    
    pixel_per_cm = side / float(grid_size_cm)
    
    return warped, M, pixel_per_cm

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Calculate pixel to cm scale based on the assumption that the detected grid/region
    # corresponds to a known physical size.
    # User mentioned "1cm x 1cm grid". 
    # If we detected the WHOLE grid, we need to know how many cells.
    # For now, we return the scale factor assuming the transformed image 
    # represents the region of interest.
    # We will need a way to know the real world size.
    # As a placeholder, we return 1.0 (needs calibration).
    pixel_per_cm = estimate_grid_pixel_per_cm(warped)
    if not _is_reasonable_pixel_per_cm(pixel_per_cm):
        plate_ppc = estimate_plate_pixel_per_cm(warped)
        if _is_reasonable_pixel_per_cm(plate_ppc):
            pixel_per_cm = float(plate_ppc)
        else:
            plate_ppc2 = estimate_plate_pixel_per_cm(image)
            if _is_reasonable_pixel_per_cm(plate_ppc2):
                pixel_per_cm = float(plate_ppc2)
            else:
                pixel_per_cm = 1.0
    
    return warped, M, pixel_per_cm

def correct_perspective(image, xml_marker_pts=None, xml_grid_size_cm=None):
    """
    Apply perspective correction.
    Priority:
    1. xml_marker_pts: User manually annotated 4-point fiducial marker. (Most accurate)
    2. Automatic detection of 4-point grid.
    3. Automatic detection of plate ellipse (generalization for 45-degree view).
    4. Fallback to original image.
    """
    
    # 1. Use XML marker if available
    if xml_marker_pts is not None and len(xml_marker_pts) == 4 and xml_grid_size_cm is not None:
        try:
            pts = np.array(xml_marker_pts, dtype="float32")
            return global_perspective_rectify(image, pts, float(xml_grid_size_cm))
        except Exception:
            pass

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
            
    if screenCnt is not None:
        warped, M, pixel_per_cm = four_point_transform(image, screenCnt.reshape(4, 2))
        return warped, M, pixel_per_cm
    
    # 3. Fallback: Check for Plate Ellipse (Generalization)
    h, w = image.shape[:2]
    M = np.eye(3, dtype="float32")
    pixel_per_cm = estimate_grid_pixel_per_cm(image)
    
    if not _is_reasonable_pixel_per_cm(pixel_per_cm):
        plate_ppc = estimate_plate_pixel_per_cm(image)
        if _is_reasonable_pixel_per_cm(plate_ppc):
            pixel_per_cm = float(plate_ppc)
        else:
            pixel_per_cm = 1.0
            
    return image, M, pixel_per_cm
