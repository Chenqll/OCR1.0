import sys

import cv2
import torch
import numpy as np

from model import CRNN
from detect_plate import Detector

CHARS_LIST = [
    "京",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "皖",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "-",
]


class CarOCR:
    # 加载模型
    def __init__(self, nhidden, model_path="./model/car_plate_crnn.pt"):
        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.nhidden = nhidden
        self.nclass = len(CHARS_LIST)
        self.loadModel(model_path)

    def loadModel(self, model_path):
        self.model = CRNN(self.nhidden, self.nclass).eval()
        self.model.to(self.DEVICE)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.DEVICE))
        self.dec = Detector()
    #图像预处理
    def pre_process(self, imgPath=""):
        self.dec.detect(imgPath)
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (224, 224))
        r, g, b = cv2.split(img)
        numpy_array = np.array([r, g, b])

        img_tensor = torch.from_numpy(numpy_array)
        img_tensor = img_tensor.float()
        img_tensor /= 256
        img_tensor = img_tensor.reshape([1, 3, 224, 224])
        img_tensor = img_tensor.to(self.DEVICE)
        return img_tensor
# 预处理后把图像输入model
    def inference(self, imgPath=""):
        image = self.pre_process(imgPath)
        output = self.model(image).cpu()
        output = torch.squeeze(output)
        _, indexs = output.max(1)
        output_label = self.post_process(indexs)
        return output_label

    def post_process(self, output):
        label = ""
        last_char = ""
        last_is_char = -1
        for i in range(output.shape[0]):
            latter = CHARS_LIST[output[i]]
            if latter == "-":
                last_is_char = 0
            else:
                if i > 0 and latter == last_char and last_is_char == 1:
                    continue
                label += latter
                last_char = latter
                last_is_char = 1
        return label


if __name__ == "__main__":
    model = CarOCR(256)
    default_path = "./temp/test.jpg"
    try:
        assert len(sys.argv) > 1
        image_path = sys.argv[1]
        output = model.inference(image_path)
        print("车牌号：{}".format(output))
        print("License Plate Recognition successfully. ")
    except AssertionError:
        print("Arguments must contain input image path.")
        print("Using the system default test image ...")
        output = model.inference(default_path)
        print("车牌号：{}".format(output))
        print("License Plate Recognition successfully. ")
    except:
        print("License Plate Recognition Failed.")
