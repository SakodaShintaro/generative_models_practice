#include "glob.hpp"
#include "vae.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char ** argv)
{
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

  VAE vae(100);
  torch::Tensor out = vae->forward(torch::randn({16, 28 * 28}));
  std::cout << out.sizes() << std::endl;
}
