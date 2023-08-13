#ifndef SAVE_IMAGE_HPP
#define SAVE_IMAGE_HPP

#include <opencv2/opencv.hpp>

#include <torch/torch.h>

void save_image(const torch::Tensor & tensor, const std::string & filename)
{
  torch::Tensor tensor_cpu = tensor.clone().cpu();
  tensor_cpu *= 255;
  tensor_cpu = tensor_cpu.to(torch::kU8);
  tensor_cpu = tensor_cpu.view({3, 32, 32});
  tensor_cpu = tensor_cpu.permute({1, 2, 0});
  tensor_cpu = tensor_cpu.contiguous();

  const int64_t height = tensor_cpu.size(0);
  const int64_t width = tensor_cpu.size(1);

  cv::Mat image(height, width, CV_8UC3);
  std::memcpy(image.data, tensor_cpu.data_ptr(), sizeof(torch::kU8) * tensor_cpu.numel());

  cv::imwrite(filename, image);
}

#endif
