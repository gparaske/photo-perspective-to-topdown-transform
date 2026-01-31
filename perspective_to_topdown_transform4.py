import cv2
import numpy as np

def show_image(window_name, image, scale=1.0):
    if scale != 1.0:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_glare(img, strength=1):
    # strength: πόσο να μειωθεί η αντανάκλαση (0 = καθόλου, 1 = πλήρης αφαίρεση) 
    img = img.astype(np.float32) / 255.0
    # Specular highlight reduction
    b, g, r = cv2.split(img)
    min_channel = np.minimum(np.minimum(r, g), b)

    specular_free = img - strength * min_channel[..., None]
    specular_free = np.clip(specular_free, 0, 1)
    return (specular_free * 255).astype(np.uint8)

def draw_directional_line(img, mean, direction, color=(0, 255, 0), length=1000, thickness=4):
    pt1 = (mean - direction * length).astype(int)
    pt2 = (mean + direction * length).astype(int)
    cv2.line(img, tuple(pt1), tuple(pt2), color, thickness)

def line_intersections_with_image(x1, y1, x2, y2, w, h):
    dx = x2 - x1
    dy = y2 - y1
    eps = 1e-9
    intersections = []
    # x = 0
    if abs(dx) > eps:
        t = (0 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= h:
            intersections.append((0, y))
    # x = w
    if abs(dx) > eps:
        t = (w - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= h:
            intersections.append((w, y))
    # y = 0
    if abs(dy) > eps:
        t = (0 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= w:
            intersections.append((x, 0))
    # y = h
    if abs(dy) > eps:
        t = (h - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= w:
            intersections.append((x, h))
    # Αν δεν βρέθηκαν τομές
    if len(intersections) == 0:
        return None, None, None, None
    # Αν βρέθηκε μόνο μία, διπλασίασέ την
    if len(intersections) == 1:
        (x3, y3) = intersections[0]
        return x3, y3, x3, y3
    # Αν βρέθηκαν περισσότερες από δύο, κράτα τις δύο πιο μακρινές
    if len(intersections) > 2:
        # Υπολόγισε αποστάσεις από το πρώτο σημείο
        p0 = np.array([x1, y1])
        intersections.sort(key=lambda p: np.linalg.norm(np.array(p) - p0))
        intersections = [intersections[0], intersections[-1]]
    (x3, y3), (x4, y4) = intersections[:2]
    return x3, y3, x4, y4

def get_topdown_board(img, model=None,debug=False):
    """
    Παίρνει ένα img (BGR).
    Επιστρέφει:
      - top_down_img: η διορθωμένη εικόνα
      - matrix: το 3x3 perspective transform matrix
    """
    if model is not None:
        # YOLO inference για ανίχνευση πλακιδίων
        from ultralytics import YOLO
        yolo_model = YOLO(model)
        results = yolo_model(img)
        r = results[0]
        if debug:
            annotated = r.plot()  # BGR
            show_image("YOLO Detections", annotated, scale=0.2)
        # boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # for (x1, y1, x2, y2) in boxes:
        #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        #     mask[y1:y2, x1:x2] = 255

    save_orig_img = img.copy()
    img = remove_glare(img, 0.9)
    h_img, w_img = img.shape[:2]
    threshold = int(min(h_img, w_img) * 0.06)  # 1/16 του μικρότερου dimension
    min_line_length = int(min(h_img, w_img) * 0.04)  # 1/25 του μικρότερου dimension
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 5, 15)
    if debug:
        show_image("Canny Edges", edges, scale=0.2)
        pass

    max_gap = 5
    while max_gap <= 20:   # upper safety limit
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_gap)
            if lines is None:
                raise ValueError("Δεν βρέθηκαν γραμμές.")

            boxes = r.boxes.xyxy.cpu().numpy() if model is not None else []
            if len(boxes) > 0:
                filtered_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    keep = False
                    for (bx1, by1, bx2, by2) in boxes:
                        if point_in_box(x1, y1, (bx1, by1, bx2, by2)) and \
                           point_in_box(x2, y2, (bx1, by1, bx2, by2)):
                            keep = True
                            break
                    if keep:
                        filtered_lines.append((x1, y1, x2, y2))
                lines = filtered_lines
            else:
                # keep original Hough format
                lines = [tuple(l[0]) for l in lines]  # convert to same format for consistency

            top, bottom, left, right = [], [], [], []
            for line in lines:
                x1, y1, x2, y2 = line#[0]
                x3, y3, x4, y4 = line_intersections_with_image(x1, y1, x2, y2, w_img, h_img)
                # TOP
                if (y3 < h_img // 2 and y4 < h_img // 2) and (x3 < w_img // 2) != (x4 < w_img // 2):
                    top.append((x1, y1, x2, y2))
                    if debug:
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1) #blue
                # BOTTOM 1/3
                elif (y3 > h_img // 2 and y4 > h_img // 2) and (x3 < w_img // 2) != (x4 < w_img // 2):
                    bottom.append((x1, y1, x2, y2))
                    if debug:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1) #yellow
                elif (x3 < w_img // 2 and x4 < w_img // 2) and (y3 < h_img // 2) != (y4 < h_img // 2):
                    left.append((x1, y1, x2, y2))
                    if debug:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1) #red
                # RIGHT
                elif (x3 > w_img // 2 and x4 > w_img // 2) and (y3 < h_img // 2) != (y4 < h_img // 2):
                    right.append((x1, y1, x2, y2))
                    if debug:
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1) #green
            if debug:
                show_image("Lines", img, 0.2)

            if len(top) == 0 or len(bottom) == 0 or len(left) == 0 or len(right) == 0:
                raise ValueError("Missing required lines: one or more of top/bottom/left/right is empty.")
            print(f"Success with maxLineGap = {max_gap}")
            break
        except Exception as e:
            print(f"Failed at maxLineGap={max_gap}: {e}")
            max_gap += 5  # increase and try again

    def line_intersection(p1, d1, p2, d2):
        A = np.array([d1, -d2]).T
        b = p2 - p1
        if np.linalg.matrix_rank(A) < 2:
            return None
        t = np.linalg.solve(A, b)
        return p1 + t[0] * d1

    def fit_line_extreme_filtered(lines, w_img, h_img, side='top', keep_n=3):
        """
        Επιλέγει την πιο αντιπροσωπευτική γραμμή με βάση:
        1. Εγγύτητα στο αντίστοιχο άκρο της εικόνας
        2. Μήκος γραμμής
        Κρατάει μόνο τις keep_n πιο ακραίες και μετά διαλέγει την πιο μεγάλη.
        """
        if not lines:
            return None
        scored = []
        for x1, y1, x2, y2 in lines:
            length = np.linalg.norm([x2 - x1, y2 - y1])
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # απόσταση από άκρο ανάλογα με την πλευρά
            if side == 'top':
                dist = my  # κοντά στο y=0
            elif side == 'bottom':
                dist = h_img - my
            elif side == 'left':
                dist = mx
            elif side == 'right':
                dist = w_img - mx
            else:
                dist = 0
            scored.append(((x1, y1, x2, y2), dist, length))
        # ταξινόμηση με βάση την εγγύτητα στο άκρο (μικρότερη dist → πιο ακραία)
        scored.sort(key=lambda x: x[1])
        # κράτα τις keep_n πιο ακραίες
        extreme_candidates = scored[:keep_n]
        # διάλεξε την πιο μεγάλη από αυτές
        best_line = max(extreme_candidates, key=lambda x: x[2])[0]
        return best_line

    x1, y1, x2, y2 = fit_line_extreme_filtered(top, w_img, h_img, side='top')
    if debug:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 15)
    m_top = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)   # μέσο σημείο
    d_top = np.array([x2 - x1, y2 - y1], dtype=np.float32)               # διεύθυνση
    x1, y1, x2, y2 = fit_line_extreme_filtered(bottom, w_img, h_img, side='bottom')
    if debug:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 15)
    m_bottom = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    d_bottom = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    x1, y1, x2, y2 = fit_line_extreme_filtered(left, w_img, h_img, side='left')
    if debug:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 15)
    m_left = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    d_left = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    x1, y1, x2, y2 = fit_line_extreme_filtered(right, w_img, h_img, side='right')
    if debug:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 15)
    m_right = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    d_right = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    if debug: draw_directional_line(img, m_top, d_top, color=(255, 0, 0))     # μπλε για πάνω
    if debug: draw_directional_line(img, m_bottom, d_bottom, color=(0, 255, 255))  # κυανό για κάτω
    if debug: draw_directional_line(img, m_left, d_left, color=(0, 0, 255))    # κόκκινο για αριστερά
    if debug: draw_directional_line(img, m_right, d_right, color=(0, 255, 0))  # πράσινο για δεξιά
    if debug: show_image("Image with Lines", img, scale=0.2)
    
    # corners από τις τομές
    tl = line_intersection(m_top, d_top, m_left, d_left)
    tr = line_intersection(m_top, d_top, m_right, d_right)
    br = line_intersection(m_bottom, d_bottom, m_right, d_right)
    bl = line_intersection(m_bottom, d_bottom, m_left, d_left)
    
    corners = [tl, tr, br, bl]
    if any(c is None for c in corners):
        raise ValueError("Δεν βρέθηκαν όλες οι γωνίες.")
    
    # Πηγή: οι πραγματικές γωνίες του board
    pts_src = np.array(corners, dtype=np.float32)
    
    # Υπολογισμός μηκών
    length_top = np.linalg.norm(tr - tl)
    length_left = np.linalg.norm(bl - tl)

    # Προορισμός: ορθογώνιο
    offset_tl = max(tl[0], tl[1])
    offset_tr = max(w_img - tr[0], tr[1])
    offset_br = max(w_img - br[0], h_img - br[1])
    offset_bl = max(bl[0], h_img - bl[1])
    offset = int(max(offset_tl, offset_tr, offset_br, offset_bl))

    new_tl = np.array([offset, offset], dtype=np.float32)
    new_tr = new_tl + np.array([length_top, 0], dtype=np.float32)
    new_bl = new_tl + np.array([0, length_left], dtype=np.float32)
    new_br = new_bl + np.array([length_top, 0], dtype=np.float32)
    pts_dst = np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)
    # Υπολογισμός canvas
    width = int(length_top + 2 * offset)
    height = int(length_left + 2 * offset)
    # Μετασχηματισμός
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    top_down_img = cv2.warpPerspective(save_orig_img, matrix, (width, height))
    return top_down_img, matrix

def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Εκτέλεση--
#img = cv2.imread("scene.jpg")
#img = cv2.imread("first_move.jpg")
img = cv2.imread("20251109_104049.jpg")

# max_gap = 5; success = False
# while max_gap <= 20:   # upper safety limit
#     try:
#         #top_down_img, M = get_topdown_board(img, maxLineGap=max_gap, model=None, debug=True)
#         top_down_img, M = get_topdown_board(img, maxLineGap=max_gap,  model=r"train_tiles_augm\best.pt", debug=True)
#         success = True
#         break
#     except Exception as e:
#         print(f"Failed at maxLineGap={max_gap}: {e}")
#         max_gap += 5  # increase and try again
# if not success:
#     print("Could not find lines with any maxLineGap.")
# else:

#top_down_img, M = get_topdown_board(img, model=None, debug=True)
top_down_img, M = get_topdown_board(img, model=r"train_tiles_augm\best.pt", debug=True)
#print(f"Success with maxLineGap = {max_gap}")
show_image("Top-down Image", top_down_img, scale=0.2)
cv2.imwrite("topdown_output.jpg", top_down_img)

