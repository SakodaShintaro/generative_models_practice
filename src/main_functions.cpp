#include "main_functions.hpp"

#include "glob.hpp"
#include "save_image.hpp"
#include "timer.hpp"
#include "vae.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <random>

constexpr int64_t kHiddenDim = 10;
const std::string kSaveDir = "./result/";
const std::string kModelName = "vae_model.pt";

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

  VAE vae(kHiddenDim);

  std::filesystem::path save_dir(kSaveDir);
  std::filesystem::remove_all(save_dir);
  std::filesystem::create_directory(save_dir);

  std::ofstream ofs(kSaveDir + "loss.tsv");
  ofs << "time\tepoch\tstep\trecon_loss\tkl_loss" << std::endl;
  ofs << std::fixed;
  std::cout << std::fixed;

  Timer timer;
  timer.start();

  torch::optim::Adam optimizer(vae->parameters(), torch::optim::AdamOptions(1e-3));

  constexpr int64_t kEpochs = 20;
  constexpr int64_t kBatchSize = 256;
  std::mt19937 engine(std::random_device{}());
  const int64_t data_num = mnist_data.size();
  torch::Device device(torch::kCUDA);
  vae->to(device);

  for (int64_t epoch = 1; epoch <= kEpochs; epoch++) {
    std::shuffle(mnist_data.begin(), mnist_data.end(), engine);
    for (int64_t i = 0; i < data_num; i += kBatchSize) {
      const int64_t step = i / kBatchSize + 1;
      std::vector<torch::Tensor> batch(
        mnist_data.begin() + i, mnist_data.begin() + std::min(i + kBatchSize, data_num));

      torch::Tensor x = torch::stack(batch, 0).to(device);
      auto [mean, var] = vae->encode(x);
      torch::Tensor z = vae->sample(mean, var);
      torch::Tensor y = vae->decode(z);
      torch::Tensor kl_loss = -0.5 * (1 + torch::log(var) - mean.pow(2) - var).sum(1).mean(0);
      torch::Tensor recon_loss =
        torch::binary_cross_entropy(y, x, {}, torch::Reduction::None).sum(1).mean(0);
      torch::Tensor loss = kl_loss + recon_loss;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      std::stringstream ss;
      ss << timer.elapsed_time() << "\t" << epoch << "\t" << step << "\t"
         << recon_loss.item<float>() << "\t" << kl_loss.item<float>() << std::endl;
      std::cout << ss.str();
      ofs << ss.str();
    }
  }

  torch::save(vae, kSaveDir + kModelName);
}

void generate(const std::string & output_dir)
{
  VAE vae(kHiddenDim);
  torch::load(vae, kSaveDir + kModelName);
  constexpr int64_t kNumSamples = 16;
  torch::Tensor z = torch::randn({kNumSamples, kHiddenDim}).to(torch::kCUDA);
  torch::Tensor y = vae->decode(z);
  for (int64_t i = 0; i < kNumSamples; i++) {
    std::stringstream ss;
    ss << output_dir << "/" << std::setfill('0') << std::setw(2) << i << ".png";
    save_image(y[i], ss.str());
  }
}
