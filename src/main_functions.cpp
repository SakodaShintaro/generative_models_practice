#include "main_functions.hpp"

#include "glob.hpp"
#include "vae.hpp"

#include <opencv2/opencv.hpp>

#include <random>

void train(const std::string & input_dir)
{
  const std::vector<std::string> files = utils::glob(input_dir);
  std::vector<torch::Tensor> mnist_data;
  for (const std::string & file : files) {
    cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
    torch::Tensor tensor =
      torch::from_blob(image.data, {image.rows * image.cols * image.channels()}, torch::kByte);
    tensor = tensor.to(torch::kFloat32);
    tensor /= 255;
    mnist_data.push_back(tensor);
  }

  VAE vae(10);

  torch::optim::Adam optimizer(vae->parameters(), torch::optim::AdamOptions(1e-3));

  constexpr int64_t kEpochs = 20;
  constexpr int64_t kBatchSize = 128;
  std::mt19937 engine(std::random_device{}());
  const int64_t data_num = mnist_data.size();
  torch::Device device(torch::kCUDA);
  vae->to(device);

  std::cout << std::fixed;

  for (int64_t epoch = 1; epoch <= kEpochs; epoch++) {
    std::shuffle(mnist_data.begin(), mnist_data.end(), engine);
    for (int64_t i = 0; i < data_num; i += kBatchSize) {
      std::vector<torch::Tensor> batch(
        mnist_data.begin() + i, mnist_data.begin() + std::min(i + kBatchSize, data_num));

      torch::Tensor x = torch::stack(batch, 0).to(device);
      auto [mean, var] = vae->encode(x);
      torch::Tensor z = vae->sample(mean, var);
      torch::Tensor y = vae->decode(z);
      torch::Tensor kl_loss = -0.5 * torch::sum(1 + torch::log(var) - mean.pow(2) - var);
      torch::Tensor recon_loss = torch::binary_cross_entropy(y, x, {}, torch::Reduction::Mean);
      torch::Tensor loss = kl_loss + recon_loss;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      std::cout << "Epoch: " << epoch << ", kl_loss: " << kl_loss.item<float>()
                << ", recon_loss: " << recon_loss.item<float>() << std::endl;
    }
  }
}

void generate()
{
}
