import numpy as np
import cv2
import math
import time

flag = -1
counter = 0


# mapping the pixels from the world plane to image plane.
def warp_perspective(frame, homography_matrix, dimension):
    frame = cv2.transpose(frame)
    dst_image = np.zeros((dimension[0], dimension[1], 3))
    for x in range(0, frame.shape[0]):
        for y in range(0, frame.shape[1]):
            TimeStamp_Homography1 = time.time()
            new_vec = np.dot(homography_matrix, [x, y, 1])
            TimeStamp_Homography2 = time.time()
            # print("TimeStamp_Homography_1:",TimeStamp_Homography2-TimeStamp_Homography1)
            new_dst_row, new_dst_col, _ = (new_vec / new_vec[2] + 0.4).astype(int)
            if new_dst_row > 3 and new_dst_row < (dimension[0] - 3):
                if new_dst_col > 3 and new_dst_col < (dimension[1] - 3):
                    dst_image[new_dst_row, new_dst_col] = frame[x, y]

                    dst_image[new_dst_row - 1, new_dst_col - 1] = frame[x, y]
                    dst_image[new_dst_row + 1, new_dst_col + 1] = frame[x, y]

                    dst_image[new_dst_row - 2, new_dst_col - 2] = frame[x, y]
                    dst_image[new_dst_row + 2, new_dst_col + 2] = frame[x, y]

            TimeStamp_Homography3 = time.time()
            # print("TimeStamp_Homography_2:", TimeStamp_Homography3 - TimeStamp_Homography2)
    # convert matrix to image
    dst_image = np.array(dst_image, dtype=np.uint8)
    dst_image = cv2.transpose(dst_image)
    return dst_image


def Test_Video(path, outputPath):
    cap1 = cv2.VideoCapture(path)
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    while (cap1.isOpened()):
        ret, img_rgb = cap1.read()
        if (ret == True):
            img, status = Generate_Lena(img_rgb)
            if (status == 1):

                out.write(img)
                if cv2.waitKey(20) and 0xFF == ord('q'):
                    break
            else:
                pass
        else:
            cap1.release()
            out.release()


def testImage(path):
    img_rgb = cv2.imread(path, 1)
    img, status = Generate_Lena(img_rgb)
    if (status == 1):
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 500, 500)
        cv2.imshow('output', img)
        cv2.waitKey(0)
    else:
        pass


def Generate_Lena(frame):
    # Find contours
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, threshold_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_perimeter_img = 90
    max_perimeter_img = 900
    flag1 = 0
    count = 0
    k = 0
    # Find parent contours and then detect tag
    for contourInfo in hierarchy[0]:
        if (contourInfo[3] == -1 and Find_Num_Children(k, hierarchy, 0) >= 2):
            max_index = k
            for i in hierarchy[0]:
                if (i[3] == max_index):
                    perimeter = cv2.arcLength(contours[count], True)
                    if (perimeter > min_perimeter_img and perimeter < max_perimeter_img):
                        flag1 = 1
                        cnt = contours[count]
                        frame, status = Project_Lena(cnt, frame)
                        count = 0
                        break
                count = count + 1
        count = 0
        k = k + 1
    if (flag1 == 1):
        return (frame, flag1)
    else:
        return (frame, 0)


def Project_Lena(cnt, frame):
    global flag
    global counter
    # Read Marker Image
    lena_img = cv2.imread('Lena.png')
    # Store reference marker's dimensions
    height, width, channel = lena_img.shape
    # Store corners in image dimensions
    x, y, w, h = cv2.boundingRect(cnt)
    # Find important contour points only
    Estimated_Contour = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
    pts1 = np.zeros([4, 2], dtype='float32')
    # check if the contour is a rectangle
    if (len(Estimated_Contour) == 4):
        n = 0

        for j in Estimated_Contour:
            if (n < 4):
                pts1[n][0] = j[0][0]
                pts1[n][1] = j[0][1]
            n += 1

        # Points of upright tag
        pts2 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        # World coordinates
        pts3 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        # Find the H matrix
        H = Find_Homography_Matrix(pts2, pts1)  # Transforming second to first

        # Make the tag upright
        TimeStamp_Warp1 = time.time()
        uprightTag = warp_perspective(frame, H, (w - 1, h - 1))
        TimeStamp_Warp2 = time.time()
        print("TimeStamp_Warp_1:", TimeStamp_Warp2 - TimeStamp_Warp1)

        # Convert image to grayscale
        grayTag = cv2.cvtColor(uprightTag, cv2.COLOR_BGR2GRAY)

        # Convert image to binary
        ret, Binarized_Tag = cv2.threshold(grayTag, 240, 255, cv2.THRESH_BINARY)

        # Smooth the edges
        Binarized_Tag = cv2.blur(Binarized_Tag, (5, 5))
        Binarized_Tag = cv2.bilateralFilter(Binarized_Tag, 5, 100, 100)

        # Detect the corners
        pts3, index = Allign_Tag( Binarized_Tag, pts2, pts3)
        pts4 = np.roll(pts2, index, axis=0)
        HForTag = Find_Homography_Matrix(pts4, pts2)

        TimeStamp_Warp3 = time.time()
        rotatedTag = warp_perspective(uprightTag, HForTag, (w - 1, h - 1))
        TimeStamp_Warp4 = time.time()
        print("TimeStamp_Warp_1:", TimeStamp_Warp4 - TimeStamp_Warp3)

        # Calculate tag ID
        tagID = Generate_TagID(rotatedTag)
        flag = tagID
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "tagID = " + str(flag), (pts1[0][0], pts1[0][1]), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        # find the Homography matrix
        H = Find_Homography_Matrix(pts1, pts3)  # Transforming second to first

        # Fit lena on to the color image
        TimeStamp_Warp5 = time.time()
        tagSizedLena = warp_perspective(lena_img, H, (frame.shape[1], frame.shape[0]))
        TimeStamp_Warp6 = time.time()
        print("TimeStamp_Warp_1:", TimeStamp_Warp6 - TimeStamp_Warp5)
        return (Draw_Image_on_Tag(frame, tagSizedLena), 1)
    else:
        return (frame, 0)


def Find_Homography_Matrix(img1, img2):
    ind = 0
    A_matrix = np.empty((8, 9))

    for pixel in range(0, len(img1)):
        x_1 = img2[pixel][0]  # Extracting pixel of world frame
        y_1 = img2[pixel][1]

        x_2 = img1[pixel][0]
        y_2 = img1[pixel][1]

        A_matrix[ind] = np.array([x_1, y_1, 1, 0, 0, 0, -x_2 * x_1, -x_2 * y_1, -x_2])
        A_matrix[ind + 1] = np.array([0, 0, 0, x_1, y_1, 1, -y_2 * x_1, -y_2 * y_1, -y_2])

        ind = ind + 2
    u, s, v = np.linalg.svd(A_matrix)
    a = []
    if v[8][8] == 1:
        for i in range(0, 9):
            a.append(v[8][i])
    else:
        for i in range(0, 9):
            a.append(v[8][i] / v[8][8])
    b = np.reshape(a, (3, 3))
    return b


def Find_Num_Children(k, hierarchy, children):
    n = 0
    # Find children of given parent
    for row in hierarchy[0]:
        if (row[3] == k):
            children = children + 1
            children = Find_Num_Children(n, hierarchy, children)
        n = n + 1
    return children


def Allign_Tag(binaryImage, points, pts1):
    # Find contours in upright tag image
    Tag_Contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find biggest contour
    areas = [cv2.contourArea(c) for c in Tag_Contours]
    max_index = np.argmax(areas)
    cnt = Tag_Contours[max_index]

    # Smooth the contour.
    Estimated_Contour = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

    # Find the corner associated with rotation
    minDistance = 0
    firstTime = 1
    index = 0
    for corners in Estimated_Contour:
        x, y = corners.ravel()

        i = 0
        for pts in points:
            i = i + 1
            border_X, border_Y = pts
            distance = math.sqrt((border_X - x) ** 2 + (border_Y - y) ** 2)
            if distance < minDistance or firstTime:
                firstTime = 0
                minDistance = distance
                index = i
    pts1 = np.roll(pts1, index - 3, axis=0)
    return (pts1, index - 3)


def Draw_Image_on_Tag(frame, lena):
    rows, cols, channels = lena.shape
    roi = frame[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask.
    img2gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Black-out area of logo in ROI
    img_black_mask = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Only bring Lena to Foreground.
    img_to_foreground = cv2.bitwise_and(lena, lena, mask=mask)
    dst_img = cv2.add(img_black_mask, img_to_foreground)

    return dst_img


def Generate_TagID(Oriented_Tag):
    # Divide image into eight parts to detect AR Tag ID
    row1 = int(Oriented_Tag.shape[0] / 8)
    col1 = int(Oriented_Tag.shape[1] / 8)
    reqRegion = np.zeros((4, 2), dtype='int32')
    reqRegion[0][0] = 3 * row1
    reqRegion[0][1] = 3 * col1
    reqRegion[3][0] = 4 * row1
    reqRegion[3][1] = 3 * col1
    reqRegion[2][0] = 4 * row1
    reqRegion[2][1] = 4 * col1
    reqRegion[1][0] = 3 * row1
    reqRegion[1][1] = 4 * col1
    lst = []
    # Check the values of the encoding region
    for i in reqRegion:
        ROI = Oriented_Tag[i[0]:i[0] + row1, i[1]:i[1] + col1]
        meanL = ROI.mean(axis=0).mean(axis=0)
        mean = meanL.sum() / 3
        if (mean > 240):
            lst.append(1)
        else:
            lst.append(0)
    ans = lst[0] * 1 + lst[1] * 2 + lst[2] * 4 + lst[3] * 8
    return ans


TimeStamp1 = time.time()
# Opening Tag0.mp4 video and saving as CubeTag0.avi
Test_Video('Tag2.mp4', 'FinalTag2.avi')
TimeStamp2 = time.time()
print("TimeStamp:", TimeStamp2 - TimeStamp1)
print("AR- TAG Detection and Lena Image Projection Complete. Please check the code directory for the generated videos.")

cv2.destroyAllWindows()
