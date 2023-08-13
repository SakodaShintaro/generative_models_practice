#ifndef SAVE_IMAGE_HPP
#define SAVE_IMAGE_HPP

#include <opencv2/opencv.hpp>

#include <torch/torch.h>

void save_image(const torch::Tensor & tensor, const std::string & filename)
{
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU);
  tensor_cpu *= 255;
  tensor_cpu = tensor_cpu.to(torch::kU8);
  tensor_cpu = tensor_cpu.view({28, 28});

  const int64_t height = tensor_cpu.size(0);
  const int64_t width = tensor_cpu.size(1);

  // OpenCVのcv::Matへの変換
  cv::Mat image(height, width, CV_8UC1);
  std::memcpy(image.data, tensor_cpu.data_ptr(), sizeof(torch::kU8) * tensor_cpu.numel());

  // 画像として保存
  cv::imwrite(filename, image);
}

#endif
