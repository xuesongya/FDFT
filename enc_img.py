import warnings
warnings.filterwarnings("ignore", module="onnxruntime")
import onnxruntime as ort
import bchlib   # 0.7
import numpy as np


# 加载ONNX模型
session = ort.InferenceSession('/home/test/backdoor/issba2/saved_models/asian_celeb/model.onnx')
# 获取输入和输出张量的名字
input1 = session.get_inputs()[0].name
input2= session.get_inputs()[1].name
output1 = session.get_outputs()[0].name
output2 = session.get_outputs()[1].name

secret = '$'

width = 112
height = 112

BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
ecc = bch.encode(data)
packet = data + ecc
# print(f'packet:{packet}{len(packet)}')

packet_binary = ''.join(format(x, '08b') for x in packet)
secret = [int(x) for x in packet_binary]
secret.extend([0, 0, 0, 0])

def encode_image(img):
    image = img.astype(np.float32) #/ 255.
    # 运行模型
    r1, r2 = session.run([output1, output2], {input1: [secret], input2:[image]})  # 运行模型并获取输出

    hidden_img = (r1[0] * 255).astype(np.uint8)
    residual = r2[0] + .5  # For visualization
    residual = np.squeeze((residual * 255).astype(np.uint8))

    return hidden_img, residual