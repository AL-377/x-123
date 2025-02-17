"""
Image utils
"""
import random
from typing import Tuple, Union, List

import cv2
import numpy as np
import rembg

from PIL import Image

import math

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(left_eye, right_eye, nose, face_pos):
    """
        check the direction and decide modify the face_pos
        calculate the rotation of face
    """

    #this function aligns given face in img based on left and right eye coordinates

    #left eye is the eye appearing on the left (right eye of the person)
    #left top point is (0, 0)

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    nose_x,nose_y = nose
    x,y,w,h = face_pos[0],face_pos[1],face_pos[2],face_pos[3]
    center_x = x + w // 2
    center_y = y + h // 2

    angle = 0
    direction = 1

    #-------------check the direction of face pos---------

    # so the face lies down
    if nose_x < min(left_eye_x,right_eye_x) or nose_x > max(left_eye_x,right_eye_y):
        x = center_x - h//2
        y = center_y - w//2
        w,h = h,w
    elif nose_y <  min(left_eye_y,right_eye_y):
        angle += 180

    new_face_pos = [x,y,w,h]

    #-------------cal the rotation-----------

    #decide the image is inverse

    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))


    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        
        #PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        #In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))
        
        
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #rotate base image

        if direction == -1:
            angle = 90 - angle

    return new_face_pos, -1 * direction * angle


def alignment_procedure_old(left_eye, right_eye,):
    """根据2只眼睛位置来进行计算人脸旋转角度"""

    # 提取左右眼睛中心点坐标
    left_eye_x, left_eye_y = left_eye[0],left_eye[1]
    right_eye_x, right_eye_y = right_eye[0],right_eye[1]

    # 计算两眼连线的斜率
    slope = (right_eye_y - left_eye_y) / (right_eye_x - left_eye_x)

    # 计算旋转角度
    angle = math.atan2(slope, 1) * 180 / math.pi

    # 计算旋转框的四个角的坐标
    # angle_rad = math.radians(angle)  # 将角度转换为弧度制
    
    return angle



def remove_padding(input_path,output_path):
    """删除图片边缘内容"""
    image = cv2.imread(input_path,cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到包含所有轮廓的最小矩形
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))

    crop_img = image[y:y+h, x:x+w]
    cv2.imwrite(output_path,crop_img,[cv2.IMWRITE_PNG_COMPRESSION, 9])
    
def remove_bg(input_path,output_path):
    """去除背景图保证大小不变"""
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = rembg.remove(input)
            o.write(output)

def pad_resize_image(
        cv2_img: np.ndarray,
        new_size: Tuple[int, int] = (640, 480),
        color: Tuple[int, int, int] = (125, 125, 125)) -> np.ndarray:
    """
    resize and pad image with color if necessary, maintaining orig scale
    args:
        cv2_img: numpy.ndarray = cv2 image
        new_size: tuple(int, int) = (width, height)
        color: tuple(int, int, int) = (B, G, R)
    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    # rescale down
    scale = min(new_w / in_w, new_h / in_h)
    # get new sacled widths and heights
    scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
    resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))
    # calculate deltas for padding
    d_w = max(new_w - scale_new_w, 0)
    d_h = max(new_h - scale_new_h, 0)
    # center image with padding on top/bottom or left/right
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    pad_resized_img = cv2.copyMakeBorder(resized_img,
                                         top, bottom, left, right,
                                         cv2.BORDER_CONSTANT,
                                         value=color)
    return pad_resized_img


def clip_coords(boxes: np.ndarray, img_shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    img1_shape: (h,w) model input shape
    img0_shape: (h,w) orig image shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xyxy2xywh(box):
    """
    Convert nx4 boxes from [x1, y1, x2, y2] to [xc, yc, w, h] where xy1=top-left, xy2=bottom-right
    """
    new_box = np.zeros_like(box)
    new_box[:, 0] = (box[:, 0] + box[:, 2]) / 2  # x center
    new_box[:, 1] = (box[:, 1] + box[:, 3]) / 2  # y center
    new_box[:, 2] = box[:, 2] - box[:, 0]  # width
    new_box[:, 3] = box[:, 3] - box[:, 1]  # height
    return new_box


def xywh2xyxy(box):
    """
    Convert nx4 boxes from [xc, yc, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
    new_box = np.zeros_like(box)
    new_box[:, 0] = box[:, 0] - box[:, 2] / 2  # top left x
    new_box[:, 1] = box[:, 1] - box[:, 3] / 2  # top left y
    new_box[:, 2] = box[:, 0] + box[:, 2] / 2  # bottom right x
    new_box[:, 3] = box[:, 1] + box[:, 3] / 2  # bottom right y
    return new_box


def draw_bbox_on_image(
        cv2_img: np.ndarray,
        boxes: Union[List[List[int]], np.ndarray],
        confs: Union[List[int], np.ndarray],
        labels: List[str],
        line_thickness: int = None,
        text_bg_alpha: float = 0.5):
    """
    Draw bboxes on cv2 image
        boxes must be 2D list/np array of coords xmin, ymin, xmax, ymax foreach bbox
        confs must be 2D list of confidences foreach corresponding bbox
    """
    img_h, img_w = cv2_img.shape[:2]
    tline = line_thickness or round(0.002 * (img_w + img_h) / 2) + 1

    for i, box in enumerate(boxes):
        label = f"{labels[i]}_{confs[i]:.2f}"
        xmin, ymin, xmax, ymax = map(int, box)
        xmin, ymin, xmax, ymax = (
            max(xmin, 0), max(ymin, 0), min(xmax,img_w), min(ymax, img_h))
        # draw bbox on image
        cv2.rectangle(cv2_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=max(
            int((img_w + img_h) / 600), 1), lineType=cv2.LINE_AA)

        # draw rect covering text
        t_size = cv2.getTextSize(
            label, 0, fontScale=tline / 3, thickness=1)[0]
        point2 = xmin + t_size[0] + 3, ymin - t_size[1] - 5
        color = (0, 0, 0)
        if text_bg_alpha == 0.0:
            cv2.rectangle(cv2_img, (xmin - 1, ymin), point2,
                          color, cv2.FILLED, cv2.LINE_AA)
        else:
            # Transparent text background
            alpha_reserve = text_bg_alpha  # 0: opaque 1: transparent
            b_c, g_c, r_c = color
            x_min, y_min = int(xmin - 1), int(ymin - t_size[1] - 3)
            x_max, y_max = int(xmin + t_size[0]), int(ymin)
            cv2_img[y_min:y_max, x_min:x_max, 0] = cv2_img[
                y_min:y_max, x_min:x_max, 0] * alpha_reserve + b_c * (1 - alpha_reserve)
            cv2_img[y_min:y_max, x_min:x_max, 1] = cv2_img[
                y_min:y_max, x_min:x_max, 1] * alpha_reserve + g_c * (1 - alpha_reserve)
            cv2_img[y_min:y_max, x_min:x_max, 2] = cv2_img[
                y_min:y_max, x_min:x_max, 2] * alpha_reserve + r_c * (1 - alpha_reserve)
        # draw label text
        cv2.putText(cv2_img, label, (xmin + 3, ymin - 4), 0, fontScale=tline / 4,
                    color=[255, 255, 255], thickness=1, lineType=cv2.LINE_AA)


def plot_one_box(bbox, img, wscale=1, hscale=1, color=None, label=None, line_thickness=None) -> None:
    """
    Plot one bounding box on image img
    args
        bboxes: bounding boxes in xyxy format (x1,y1,x2,y2)
        img: image in (H,W,C) numpy.ndarray fmt
        wscale: multiplication factor for width (default 1 if no scaling required)
        hscale: multiplication factor for height (default 1 if no scaling required)
    """
    tline = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    coord1 = (int(bbox[0] * wscale), int(bbox[1] * hscale))
    coord2 = (int(bbox[2] * wscale), int(bbox[3] * hscale))
    cv2.rectangle(img, coord1, coord2, color, thickness=tline, lineType=cv2.LINE_AA)
    if label:
        tfont = max(tline - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tline / 3, thickness=tfont)[0]
        coord2 = coord1[0] + t_size[0], coord1[1] - t_size[1] - 3
        cv2.rectangle(img, coord1, coord2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (coord1[0], coord1[1] - 2), 0, tline / 3,
                    [225, 255, 255], thickness=tfont, lineType=cv2.LINE_AA)


def resize_maintaining_aspect(img, width, height):
    """
    If width and height are both None, no resize is done
    If either width or height is None, resize maintaining aspect
    """
    old_h, old_w, _ = img.shape

    if width is not None and height is not None:
        new_w, new_h = width, height
    elif width is None and height is not None:
        new_h = height
        new_w = (old_w * new_h) // old_h
    elif width is not None and height is None:
        new_w = width
        new_h = (new_w * old_h) // old_w
    else:
        # no resizing done if both width and height are None
        return img
    img = cv2.resize(img, (new_w, new_h))
    return img
