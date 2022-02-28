import sys
import cv2
import copy
import torch

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression_plate, scale_coords, xyxy2xywh


class Detector:
    def __init__(self, model_path="./model/detector.pt"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path, self.device)

    def load_model(self, weights, device):
        self.model = attempt_load(
            weights, map_location=device)  # load FP32 model

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
        coords[:, :8] /= gain
        #clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4

        return coords

    def save_result(self, img, xywh, image_path):
        h, w, _ = img.shape

        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

        img0 = img[y1:y2, x1:x2]
        cv2.imwrite(image_path, img0)

    def detect(self, image_path):
        # Load model
        img_size = 800
        conf_thres = 0.3
        iou_thres = 0.5

        orgimg = cv2.imread(image_path)  # BGR
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + image_path
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)),
                              interpolation=interp)

        imgsz = check_img_size(
            img_size, s=self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression_plate(pred, conf_thres, iou_thres)
        det = pred[0]
        # Process detections
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(
            self.device)  # normalization gain whwh

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = self.scale_coords_landmarks(
                img.shape[2:], det[:, 5:13], orgimg.shape).round()

            xywh = (xyxy2xywh(det[0, :4].view(1, 4)) / gn).view(-1).tolist()
            self.save_result(orgimg, xywh, image_path)


if __name__ == '__main__':
    dec = Detector()
    default_path = "./temp/test.jpg"
    try:
        assert len(sys.argv) > 1
        image_path = sys.argv[1]
        dec.detect(image_path)
        print('License plate detection success!')
    except AssertionError:
        print("Arguments must contain input image path.")
        print("Using the system default test image ...")
        dec.detect(default_path)
        print('License plate detection success!')
    except:
        print('License plate detection Failed!')
