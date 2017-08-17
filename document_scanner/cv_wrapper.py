# import the necessary packages
import cv2
import numpy as np

# canny edge method, not currently used
def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
    gray = cv2.GaussianBlur(gray, (17, 17), 0) # blur
    gray = cv2.bilateralFilter(gray,9,75,75) # what does this do again?
    edged = cv2.Canny(gray, 100, 200, apertureSize=5) # find edges

    # Blur, erode, dilate to remove noise. May have to look at net-effect
    edged = cv2.GaussianBlur(edged, (17, 17), 0)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.erode(edged, kernel, iterations=1)
    edged = cv2.dilate(edged, kernel, iterations=1)
    return edged

def get_keypoints_and_descriptors(img, orb, n=3):
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

    return keypoints, np.concatenate(descriptors)

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


def resize(img, height):
    h, w = img.shape[:2]
    ratio = height/h
    return cv2.resize(img, (int(ratio*w), height))


def display(*imgs):
    for img in imgs:
        cv2.imshow('', resize(img, 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_matching_points(des_template, des_photo):

    flann_index_params = get_flann_index_params()
    flann_search_params = dict(checks=50)  # 50-100

    flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    matches = flann.knnMatch(des_template, des_photo, k=2)

    return matches


def select_good_matches(matches):
    # remove matches that are likely to be incorrect, using Lowe's ratio test
    return [m for (m, n) in matches if m.distance < 0.7 * n.distance]  # 0.65-0.8, false negatives-false positives


def find_transformation_and_mask(kp_template, kp_photo, matches):

    src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_photo[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # use RANSAC method to discount suspect matches
    transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #transform = cv2.estimateRigidTransform(src_pts, dst_pts, True)

    return transform, mask


def reverse_transformation(photo, transform, original_shape):
    # inverse the transformation to retrieve the original
    h, w = original_shape[:2]
    return cv2.warpPerspective(photo, np.linalg.inv(transform), (w, h))


def pad_coords(coords, padding):
    l, r, u, d = coords
    return (l - padding, r + padding, u - padding, d + padding)


def crop_section(image, coords):
    l, r, u, d = coords
    return image[u:d, l:r]


def crop_sections(image, df_with_coords):
    df_with_coords['crop'] = df_with_coords['coords'].apply(lambda x: crop_section(image, x))
    return df_with_coords