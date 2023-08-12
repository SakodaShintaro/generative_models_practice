#include "glob.hpp"

#include <opencv2/opencv.hpp>

#include <torch/torch.h>

#include <iostream>

int main(int argc, char ** argv)
{
  std::cout << "vae_cpp" << std::endl;

  if (argc != 2) {
    std::cout << "Usage: vae_cpp <input_dir>" << std::endl;
    return 1;
  }

  const std::string input_dir = argv[1];
  const std::vector<std::string> files = utils::glob(input_dir);
  for (const std::string & file : files) {
    cv::Mat image = cv::imread(file);
    torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, 1}, torch::kByte);
    std::cout << file << " " << image.size() << " " << tensor.sizes() << std::endl;
  }
}
