import dlib
import math
import skimage
from skimage import io, transform
import cv2
import numpy as np


def detect_idcard_area(pic):
    """
    This part use dlib to detect the 5-points face landmark.you can use openface for much more accuracy landmark localization effect.
    :param pic:
    :return:
    """
    # load model
    predictor_path = './shape_predictor_5_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # detect face
    img = io.imread(pic)
    dets = detector(img, 1)
    assert len(dets) > 0
    d = dets[0]
    # localize landmark of face
    shape = predictor(img, d).parts()
    det_x = shape[0].x - shape[2].x
    det_y = shape[0].y - shape[2].y
    # calculate the rotate angle
    angle = math.degrees(math.atan(det_y / det_x))
    # rotate image
    rotated_img = transform.rotate(img, angle)
    # you can save the intermediate result
    io.imsave('rotated_img.png', rotated_img)
    rotated_img = io.imread('rotated_img.png')
    # detect the rotated image to get the portrait location
    dets = detector(rotated_img, 1)
    assert len(dets) > 0
    d = dets[0]
    left, top, portrait_width, portrait_height = d.left(), d.top(), d.width(), d.height()
    # calculate the relative coordinate
    # the chinese id card's width is 85.6mm,height is 54.0mm,the portrait area width is 26mm,height is 32mm,
    # the relative coordinate of the portrait left top corner is x in range of [0.65,0.75],y in range of [0.15,0.2]
    # you can adjust the parameter for specific use
    face_portrait_height_ratio = 0.3
    face_portrait_width_ratio = 0.6
    portrait_card_relative_x_distance = 0.7
    portrait_card_relative_y_distance = 0.35
    # calculate the coordinate of the card,and crop it to refine
    card_height = portrait_height / face_portrait_height_ratio / 32 * 60
    card_width = portrait_width / face_portrait_width_ratio / 26 * 95
    card_left_top_x = left - portrait_card_relative_x_distance * card_width
    card_left_top_y = top - portrait_card_relative_y_distance * card_height
    cropped_image = rotated_img[int(card_left_top_y):int(card_left_top_y + card_height),
                    int(card_left_top_x):int(card_left_top_x + card_width), :]
    io.imsave('cropped_image.png', cropped_image)
    cv2.imshow('cropped_image', cropped_image)
    cv2.waitKey(0)


def refine_card_area(pic):
    """
    This part try to refine the idcard area with opencv,so that the ER part will benifits from the alignment of card,the accuracy will improve a lot.
    :param pic:
    :return:
    """
    width = 640
    height = 480
    fixed_width = int(width * 0.73)
    fixed_height = int(height * 0.625)
    coordinate = [[fixed_width, fixed_height], [fixed_width, 0], [], [0, fixed_height], [0, 0]]
    image = io.imread(pic)
    ms = cv2.pyrMeanShiftFiltering(image, 8, 20)
    canny = cv2.Canny(ms, 30, 10)
    cv2.imshow('canny',canny)
    cv2.waitKey(0)
    # the point follow the Counter clockwise
    _, cnts, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    for j in range(len(cnts)):
        c = cnts[j]
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        # we assert the available area ratio bigger than 0.5
        if len(approx) == 4 and cv2.contourArea(approx.squeeze(1)) / (fixed_height * fixed_height) >= 0.5:
            compute_direction = [0, 0, 0, 0]
            for i in range(4):
                delta = approx[(i + 1) % 4, 0, :] - approx[i, 0, :]
                delta_argmax = int(abs(delta[1]) > abs(delta[0]))
                # compute the line direction
                compute_direction[i] = (
                    (delta_argmax + 1) * (1 if delta[delta_argmax] > -delta[delta_argmax] else -1) + 2)
            pts1 = approx.astype(np.float32)
            # compute the target warp coordinate
            pts2 = np.array([coordinate[x] for x in compute_direction], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warp = cv2.warpPerspective(image, M, (fixed_width, fixed_height))
            io.imsave('warped.png', warp)
            cv2.imshow('warped', warp)
            cv2.waitKey(0)


def detect_text_localization_1(pic):
    """
    use a pre-trained ER filter to classify the area whether contains text line.
    """
    image = cv2.imread(pic)
    vis = np.array(image)
    erc2 = cv2.text.loadClassifierNM2('./trained_classifierNM2.xml')
    erc1 = cv2.text.loadClassifierNM1('./trained_classifierNM1.xml')
    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(image)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))

    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    for channel in channels:
        er1 = cv2.text.createERFilterNM1(erc1, 12, 0.00015, 0.15, 0.5, True, 0.3)
        er2 = cv2.text.createERFilterNM2(erc2, 0.3)
        regions = cv2.text.detectRegions(channel, er1, er2)
        if len(regions) == 0:
            continue
        rects = cv2.text.erGrouping(image, channel, [r.tolist() for r in regions])

        # Visualization
        for r in range(0, np.shape(rects)[0]):
            rect = rects[r]
            print((rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]))
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
    cv2.imshow('vis', vis)
    cv2.waitKey(0)


def detect_text_localization_2(pic):
    import sys, os
    import tensorflow as tf
    sys.path.append(os.path.join(os.getcwd(), 'text-detection-ctpn'))
    from ctpnlib.networks.factory import get_network
    from ctpnlib.fast_rcnn.config import cfg
    from ctpnlib.fast_rcnn.test import test_ctpn
    from ctpnlib.fast_rcnn.nms_wrapper import nms
    from ctpnlib.utils.timer import Timer
    from text_proposal_connector import TextProposalConnector

    def connect_proposal(text_proposals, scores, im_size):
        cp = TextProposalConnector()
        line = cp.get_text_lines(text_proposals, scores, im_size)
        return line

    def save_results(image_name, im, line, thresh):
        inds = np.where(line[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = line[i, :4]
            score = line[i, -1]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)
        image_name = image_name.split('/')[-1]
        cv2.imwrite('detected_' + image_name, im)

    def ctpn(sess, net, image_name):
        img = cv2.imread(image_name)
        im = np.array(img)
        timer = Timer()
        timer.tic()
        scores, boxes = test_ctpn(sess, net, im)
        sess.close()
        timer.toc()
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

        # Visualize detections for each class
        CONF_THRESH = 0.9
        NMS_THRESH = 0.3
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        keep = np.where(dets[:, 4] >= 0.7)[0]
        dets = dets[keep, :]
        line = connect_proposal(dets[:, 0:4], dets[:, 4], im.shape)
        save_results(image_name, im, line, thresh=0.9)
        return line

    # Use RPN for proposals
    cfg.TEST.HAS_RPN = True
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, './VGGnet_fast_rcnn_iter_50000.ckpt')
    return ctpn(sess, net, pic)


def clear_noise(image, n, z):
    height, width = image.shape
    new_image = np.copy(image)
    for _ in range(z):
        new_image = np.copy(new_image)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if image[y, x] == 1:
                    count = image[y - 1, x - 1] + image[y, x - 1] + image[y + 1, x - 1] + image[y - 1, x] + image[
                        y + 1, x] + image[y - 1, x + 1] + image[y, x + 1] + image[y + 1, x + 1]
                    if count < n:
                        new_image[y, x] = 0

    return new_image


def get_net(num_outputs):
    from mxnet import gluon
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=0, strides=1))
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=0, strides=1))
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=0, strides=1))
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(num_outputs))
    return net


def character_recognize(pic, text_area):
    img = cv2.imread(pic)
    mser = cv2.MSER_create()
    import mxnet.ndarray as nd
    import mxnet as mx
    with open('chinese.txt') as to_read:
        chinese = [m_line.strip() for m_line in to_read]
    net = get_net(len(chinese))
    ctx = mx.gpu()
    net.load_params('./single_character_recognition/train_model/chinese_2.para',ctx)

    for m_text_area in text_area:
        point1_x, point1_y, point2_x, point2_y = m_text_area[0], m_text_area[1], m_text_area[2], m_text_area[3]
        # roi of text area
        pic_text_area = img[int(point1_y):int(point2_y), int(point1_x):int(point2_x), :]

        gray = cv2.cvtColor(pic_text_area, cv2.COLOR_RGB2GRAY)
        binaray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,5)
        regions, boxes = mser.detectRegions(gray)
        to_predict_pics = []
        boxes = sorted(boxes,key=lambda x:x[0])
        if len(boxes) == 0:
            continue
        # remove the similar boxes from redundant boxes
        refined_boxes = [boxes[0]]
        superposition_ratio = .5
        for i in range(1,len(boxes)):
            last_x,last_width = boxes[i-1][0],boxes[i-1][2]
            cur_x,cur_width = boxes[i][0],boxes[i][2]
            total_len = cur_x+cur_width-last_x
            share_len = last_width-cur_x+last_x
            if share_len/total_len < superposition_ratio:
                refined_boxes.append(boxes[i])
        for box in refined_boxes:
            x, y, w, h = box
            if w / pic_text_area.shape[1] <= 0.9 and w / h < 1.5:
                char_pic = cv2.resize(binaray[y:y + h, x:x + w], (28, 28))
                to_predict_pics.append(char_pic/255)
        if len(to_predict_pics) == 0:
            continue
        output = net(nd.array(to_predict_pics).reshape((-1,1,28,28)).as_in_context(ctx))
        predictions = nd.argmax(output, axis=1).asnumpy()
        predictions_char = [chinese[int(m_prediction)] for m_prediction in predictions]
        print(' '.join(predictions_char))
        cv2.imshow('to_recognize',pic_text_area)
        cv2.waitKey(0)

if __name__ == '__main__':

    # detect_idcard_area('./test.jpg')
    # refine_card_area('./cropped_image.png')
    # text_area = detect_text_localization_2('warped.png')
    #
    #
    # # sort text_area by the y-axis
    # def sort_text_area(x):
    #     return x[1]
    #
    #
    # text_area = sorted(text_area, key=sort_text_area)
    # with open('text_area.txt','w') as to_write:
    #     for m_text_area in text_area:
    #         to_write.write(' '.join(map(lambda x:str(x),m_text_area[:4]))+'\n')
    text_area = []
    with open('text_area.txt') as to_read:
        for m_line in to_read:
            text_area.append([float(x) for x in m_line.strip().split(' ')])
    character_recognize('warped.png', text_area)
    pass
