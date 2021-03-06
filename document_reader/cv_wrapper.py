# import the necessary packages
import cv2
import math
import numpy as np

# canny edge method, not currently used
def find_edges(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    grey = cv2.GaussianBlur(grey, (17, 17), 0) # blur
    grey = cv2.bilateralFilter(grey, 9, 75, 75) # what does this do again?
    edged = cv2.Canny(grey, 100, 200, apertureSize=5) # find edges

    # Blur, erode, dilate to remove noise. May have to look at net-effect
    edged = cv2.GaussianBlur(edged, (17, 17), 0)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.erode(edged, kernel, iterations=1)
    edged = cv2.dilate(edged, kernel, iterations=1)
    return edged


def get_keypoints_and_descriptors(img, orb, n=3):
    #img = find_edges_2(img)
    # Gets keypoints from all parts of the image by subdividing it into n*n parts and calculating keypoints separately
    keypoints = []
    descriptors = []
    h,w = img.shape[:2]
    for i in range(0,n):
        for j in range(0,n):
            mask = np.zeros((h, w), dtype = 'uint8')
            cv2.rectangle(mask, (i*w//n, j*h//n), ((i+1)*w//n, (j+1)*h//n), 255, cv2.FILLED)
            kp, des = orb.detectAndCompute(img, mask)
            if kp is not None:
                keypoints.extend(kp)
            # descriptors need to be in a numpy array
            if des is not None:
                descriptors.append(des)

    return keypoints, np.concatenate(descriptors) if descriptors else None


def get_orb(n=3):
    # Initiate ORB detector
    return cv2.ORB_create(nfeatures=12000 // (n * n),  # find many features
                         patchSize=31,  # granularity
                         edgeThreshold=1)  # edge margin to ignore


def get_flann_index_params():
    FLANN_INDEX_LSH = 6

    return dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,  # 6-12
                              key_size=12,  # 12-20
                              multi_probe_level=1)  # 1-2


def reshape(img, shape):
    h, w, c = shape
    if len(img.shape) == 2 and c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if h or w:
        h = h or img.shape[0]
        w = w or img.shape[1]
        img = cv2.resize(img, (w, h))
    return img


def resize(img, length):
    if not length:
        return img
    h, w = img.shape[:2]
    new_height, new_width = (length, int((length/h)*w)) if h > w else (int((length/w)*h), length)
    return cv2.resize(img, (new_width, new_height))


def display(*imgs):
    for img in imgs:
        cv2.imshow('', resize(img, 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_matching_points(des_template, des_photo):

    flann_index_params = get_flann_index_params()
    flann_search_params = dict(checks=50)  # 50-100
    flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    matches = flann_matcher.knnMatch(des_template, des_photo, k=2)
    
    # Lowe's ratio test, removes matches that are likely to be incorrect
    if matches is not None:
        lowe_ratio = 0.7 # 0.65-0.8, false negatives-false positives
        matches = [m[0] for m in matches if len(m) == 1 or (len(m) >= 2 and m[0].distance < lowe_ratio * m[1].distance)]
    return matches
        
        
def get_distance(x, y):
    return math.hypot(x[0] - y[0], x[1] - y[1])


def find_transform_and_mask(kp_template, kp_photo, matches):

    src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # use RANSAC method to discount suspect matches
    transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return transform, mask


def reverse_transformation(photo, transform, original_shape):
    # inverse the transformation to retrieve the original
    h, w = original_shape[:2]
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.linalg.LinAlgError as err:
        return None
    return cv2.warpPerspective(photo, inverse, (w, h))


def pad_lrud(lrud, padding):
    l, r, u, d = lrud
    return l - padding, r + padding, u - padding, d + padding


def crop_section(image, lrud):
    l, r, u, d = lrud
    l, u = max(0, l), max(0, u)
    return image[u:d, l:r]


def crop_sections(image, df_with_lrud):
    df_with_crops = df_with_lrud.copy()
    df_with_crops['crop'] = df_with_crops['lrud'].apply(lambda x: crop_section(image, x))
    return df_with_crops


def low_pass_filter(img):
    # subtract effect of low-pass filter (convolution with 5x5 Gaussian kernel)
    return img + (img - cv2.GaussianBlur(img, (5, 5), 0))


def high_pass_filter(img):
    # high-pass filter (convolution with 3x3 kernel that approximates Laplacean)
    return cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


def low_and_high_pass_filter(img):
    return high_pass_filter(low_pass_filter(img))
