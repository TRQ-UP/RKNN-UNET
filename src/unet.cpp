#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "rknn_api.h"
#include <chrono>

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



int main() {
    auto start = std::chrono::high_resolution_clock::now();
    const char* img_path = "/home/ubuntu/npu_test/unet/img/01_test.tif";
    const char* roi_mask_path = "/home/ubuntu/npu_test/unet/img/01_test_mask.png";
    const char *model_path =  "/home/ubuntu/npu_test/unet/model/eyes_unet-sim-3588.rknn";


    // Load ROI mask
    Mat roi_img = imread(roi_mask_path, IMREAD_GRAYSCALE);
    if (roi_img.empty()) {
        cout << "Image not found: " << roi_mask_path << endl;
        return -1;
    }

    // Load image
    Mat original_img = imread(img_path);
    if (original_img.empty()) {
         cout << "Image not found: " << img_path << endl;
        return -1;
    }

    // Convert image to RGB
    cvtColor(original_img, original_img, COLOR_BGR2RGB);

    // Expand batch dimension
    // Mat img = original_img.reshape(1, 1);
    Mat img = original_img;


    const int MODEL_IN_WIDTH = 565;
    const int MODEL_IN_HEIGHT = 584;
    const int MODEL_IN_CHANNELS = 3;


    rknn_context ctx = 0;
    int ret;
    int model_len = 0;
    unsigned char *model;

  // ======================= 初始化RKNN模型 ===================
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    if (ret < 0)
    {
      printf("rknn_init fail! ret=%d\n", ret);
      return -1;
    }


  // ======================= 获取模型输入输出信息 ===================
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }

    // ======================= 设置模型输入 ===================
    // 使用rknn_input结构体存储模型输入信息, 表示模型的一个数据输入,用来作为参数传入给 rknn_inputs_set 函数
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;                                                     // 设置模型输入索引
    inputs[0].type = RKNN_TENSOR_UINT8;                                      // 设置模型输入类型
    inputs[0].size = img.cols * img.rows * img.channels() * sizeof(uint8_t); // 设置模型输入大小
    inputs[0].fmt = RKNN_TENSOR_NHWC;                                        // 设置模型输入格式：NHWC
    inputs[0].buf = img.data;                                                // 设置模型输入数据

    // 使用rknn_inputs_set函数设置模型输入
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
      printf("rknn_input_set fail! ret=%d\n", ret);
      return -1;
    }

    // ======================= 推理 ===================
    printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
      printf("rknn_run fail! ret=%d\n", ret);
      return -1;
    }

    // ======================= 获取模型输出 ===================
    // 使用rknn_output结构体存储模型输出信息
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    // 使用rknn_outputs_get函数获取模型输出
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
      printf("rknn_outputs_get fail! ret=%d\n", ret);
      return -1;
    }

    float *output_data = (float *)outputs[0].buf;
    int output_size = outputs[0].size / sizeof(uint32_t);
    // cout << "channel: " << channel << endl;
    // cout << "res: " << res << endl;
    // cout << "size: " << output_size << endl;
    int img_h = 584;
    int img_w = 565;
    int channel = output_size / img_h / img_w;
    int res = output_size % (img_h * img_w);
    cout << "channel: " << channel << endl;
    cout << "res: " << res << endl;
    int* label_data = new int[img_h * img_w];
    if (res != 0)
    {
        fprintf(stderr, "output shape is not supported.\n");
    }
    else
    {
      /* multi-class segmentation */
      for (int i = 0; i < img_h; ++i)
      {
          for (int j = 0; j < img_w; ++j)
          {
              int argmax_id = -1;
              float max_conf = std::numeric_limits<float>::min();
              for (int k = 0; k < channel; ++k)
              {
                  float out_value = output_data[k * img_w * img_h + i * img_w + j];
                  if (out_value > max_conf)
                  {
                      argmax_id = k;
                      max_conf = out_value;
                  }
              }
              label_data[i * img_w + j] = argmax_id;
              if (label_data[i * img_w + j] == 1) {
                label_data[i * img_w + j] = 255;
            }
          }
      }
    }

    // 将图像数据存储到一维数组中
    int* roi_array = new int[img_h * img_w];
    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            roi_array[i * img_w + j] = static_cast<int>(roi_img.at<uchar>(i, j));
        }
    }

    for (int i = 0; i < img_h; ++i) {
      for (int j = 0; j < img_w; ++j) {
        if (roi_array[i * img_w + j] == 0) {
                label_data[i * img_w + j] = roi_array[i * img_w + j];
            }
        }
    }
    Mat result_img(img_h, img_w, CV_8UC1);
    for (int i = 0; i < img_h; ++i) {
        for (int j = 0; j < img_w; ++j) {
            result_img.at<uchar>(i, j) = static_cast<uchar>(label_data[i * img_w + j]);
        }
    }
    imwrite("result.png", result_img);


    // Release resources
    rknn_outputs_release(ctx, 1, outputs);
    if (ret < 0)
    {
      printf("rknn_outputs_release fail! ret=%d\n", ret);
      return -1;
    }
    else if (ctx > 0)
    {
      // ======================= 释放RKNN模型 ===================
      rknn_destroy(ctx);
    }
    // ======================= 释放模型数据 ===================
    if (model)
    {
      free(model);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}
