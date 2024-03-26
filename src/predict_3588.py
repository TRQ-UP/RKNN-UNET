import os
import time
import numpy as np
from PIL import Image
from rknnlite.api import RKNNLite


def main():
    t_start = time.time()
    # classes = 1  # exclude background
    RKNN_MODEL = "../model/eyes_unet-sim-3588.rknn"
    img_path = "../img/01_test.tif"
    roi_mask_path = "../img/01_test_mask.gif"
    # assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."


    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img.save("mask.png")
    roi_img = np.array(roi_img)
    


    # load image
    original_img = Image.open(img_path).convert('RGB')
    # from pil image to tensor and normalize
    # data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    # img = data_transform(original_img)
    # expand batch dimension
    img = np.array(original_img)
    img = img[np.newaxis, :]


    # Create RKNN object
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)


    # Init runtime environment
    print('--> Init runtime environment')

    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    output = rknn_lite.inference(inputs=[img])
    # print("output:",output)
    # t_end = time.time()
    # print("inference time: {}".format(t_end - t_start))
    output = np.array(output).reshape(1, 2, 584, 565)
    prediction = np.squeeze(np.argmax(output, axis=1))
    print(prediction.shape)


    prediction = prediction.astype(np.uint8)
    # np.save("int8_unet.npy", prediction)
    # 将前景对应的像素值改成255(白色)
    prediction[prediction == 1] = 255
    # 将不敢兴趣的区域像素设置成0(黑色)
    prediction[roi_img == 0] = 0
    mask = Image.fromarray(prediction)
    mask.save("test_result_3588.png")

    rknn_lite.release()
    print("output:",output)
    t_end = time.time()
    print("inference time: {}".format(t_end - t_start))

if __name__ == '__main__':
    main()
