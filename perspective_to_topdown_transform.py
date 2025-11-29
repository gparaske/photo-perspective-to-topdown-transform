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

def draw_directional_line(img, mean, direction, color=(0, 255, 0), length=1000, thickness=4):
    pt1 = (mean - direction * length).astype(int)
    pt2 = (mean + direction * length).astype(int)
    cv2.line(img, tuple(pt1), tuple(pt2), color, thickness)

def get_topdown_board(img, debug=False):
    """
    Παίρνει ένα img (BGR).
    Επιστρέφει:
      - top_down_img: η διορθωμένη εικόνα
      - matrix: το 3x3 perspective transform matrix
    """
    h_img, w_img = img.shape[:2]
    threshold = int(min(h_img, w_img) * 0.06)  # 1/16 του μικρότερου dimension
    min_line_length = int(min(h_img, w_img) * 0.04)  # 1/25 του μικρότερου dimension
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    if debug:
        show_image("Canny Edges", edges, scale=0.2)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=10)
    if lines is None:
        raise ValueError("Δεν βρέθηκαν γραμμές.")

    top, bottom, left, right = [], [], [], []

    cam_angle = 45  # μοίρες
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < cam_angle:
            if y1 < h_img // 2 and y2 < h_img // 2:
                top.append((x1, y1, x2, y2))
                if debug:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            elif y1 > h_img // 2 and y2 > h_img // 2:
                bottom.append((x1, y1, x2, y2))
                if debug:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
        elif abs(angle - 90) < cam_angle or abs(angle + 90) < cam_angle:
            if x1 < w_img // 2 and x2 < w_img // 2:
                left.append((x1, y1, x2, y2))
                if debug:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            elif x1 > w_img // 2 and x2 > w_img // 2:
                right.append((x1, y1, x2, y2))
                if debug:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

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
    top_down_img = cv2.warpPerspective(img, matrix, (width, height))
    return top_down_img, matrix

# Εκτέλεση--
img = cv2.imread("first_move.jpg")
show_image("Original Image", img, scale=0.2)
top_down_img, M = get_topdown_board(img, debug=True)
show_image("Top-down Image", top_down_img, scale=0.2)
