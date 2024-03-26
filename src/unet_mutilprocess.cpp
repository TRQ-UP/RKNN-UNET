#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "rknn_api.h"
#include <chrono>
#include <thread>
#include <mutex>
using namespace std;
using namespace cv;


static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp = fopen(filename, "rb");
  if (fp == nullptr)
  {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int model_len = ftell(fp);
  unsigned char *model = (unsigned char *)malloc(model_len); // 申请模型大小的内存，返回指针
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp))
  {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp)
  {
    fclose(fp);
  }
  return model;
}

// 图像前处理模块
Mat preprocessImage(const char* img_path) {
    Mat img = imread(img_path);
    if (img.empty()) {
        cout << "Image not found: " << img_path << endl;
        return Mat();
    }

    // Convert image to RGB
    cvtColor(img, img, COLOR_BGR2RGB);

    return img;
}

// 数据后处理模块
Mat postprocessImage(const Mat& label_data, const Mat& roi_img) {
    int img_h = label_data.rows;
    int img_w = label_data.cols;
    Mat result_img(img_h, img_w, CV_8UC1);

    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            if (roi_img.at<uchar>(i, j) == 0) {
                result_img.at<uchar>(i, j) = roi_img.at<uchar>(i, j);
            } else {
                result_img.at<uchar>(i, j) = label_data.at<uchar>(i, j);
            }
        }
    }

    return result_img;
}

void initModel(const char* model_path, rknn_context* ctx) {
    int ret;
    int model_len = 0;
    unsigned char *model;

    // Load model
    model = load_model(model_path, &model_len);
    ret = rknn_init(ctx, model, model_len, 0, NULL);
    if (ret < 0) {
        cout << "Model initialization failed!" << endl;
    }

    free(model);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    const char* img_path = "/home/ubuntu/npu_test/unet/img/01_test.tif";
    const char* roi_mask_path = "/home/ubuntu/npu_test/unet/img/01_test_mask.png";
    const char *model_path =  "/home/ubuntu/npu_test/unet/model/eyes_unet-sim-3588.rknn";

    // 初始化RKNN模型
    rknn_context ctx = 0;
    std::thread modelInitThread(initModel, model_path, &ctx);

    // 图像前处理
    Mat roi_img = imread(roi_mask_path, IMREAD_GRAYSCALE);
    if (roi_img.empty()) {
        cout << "Image not found: " << roi_mask_path << endl;
        return -1;
    }

    // 图像前处理
    Mat original_img = preprocessImage(img_path);
    if (original_img.empty()) {
        return -1;
    }

    const int MODEL_IN_WIDTH = 565;
    const int MODEL_IN_HEIGHT = 584;
    const int MODEL_IN_CHANNELS = 3;

    // 设置模型输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = original_img.cols * original_img.rows * original_img.channels() * sizeof(uint8_t);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = original_img.data;

    // 等待模型初始化完成
    modelInitThread.join();

    // 推理
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        return -1;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        return -1;
    }

    // 获取模型输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        return -1;
    }

    float *output_data = (float *)outputs[0].buf;
    int output_size = outputs[0].size / sizeof(uint32_t);
    int img_h = 584;
    int img_w = 565;
    int channel = output_size / (img_h * img_w);
    int res = output_size % (img_h * img_w);

    // 数据后处理
    if (res != 0) {
        fprintf(stderr, "output shape is not supported.\n");
    } else {
        Mat label_data(img_h, img_w, CV_8UC1, Scalar(0));
        for (int i = 0; i < img_h; ++i) {
            for (int j = 0; j < img_w; ++j) {
                int argmax_id = 0;
                float max_conf = output_data[i * img_w + j];
                for (int k = 1; k < channel; ++k) {
                    float out_value = output_data[k * img_w * img_h + i * img_w + j];
                    if (out_value > max_conf) {
                        argmax_id = k;
                        max_conf = out_value;
                    }
                }
                label_data.at<uchar>(i, j) = (argmax_id == 1) ? 255 : argmax_id;
            }
        }

        Mat result_img = postprocessImage(label_data, roi_img);
        imwrite("result.png", result_img);
    }

    // Release resources
    rknn_outputs_release(ctx, 1, outputs);
    if (ctx > 0) {
        rknn_destroy(ctx);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
