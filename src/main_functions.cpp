#include "main_functions.hpp"

#include "glob.hpp"
#include "save_image.hpp"
#include "timer.hpp"
#include "vae.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <random>

constexpr int64_t kHiddenDim = 30;
const std::string kSaveDir = "./result/";
const std::string kModelName = "vae_model.pt";

void train(const std::string & input_dir)
{
  const std::vector<std::string> files = utils::glob(input_dir);
  std::vector<torch::Tensor> data_vector;
  for (const std::string & file : files) {
    cv::Mat image = cv::imread(file);
    cv::resize(image, image, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
    torch::Tensor tensor =
      torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kByte);
    tensor = tensor.to(torch::kFloat32);
    tensor /= 255;
    tensor = tensor.permute({2, 0, 1});
    tensor = tensor.contiguous();
    data_vector.push_back(tensor);
  }

  VAE vae(kHiddenDim);

  std::filesystem::path save_dir(kSaveDir);
  std::filesystem::remove_all(save_dir);
  std::filesystem::create_directory(save_dir);
  std::filesystem::create_directory(save_dir / "images");

  std::ofstream ofs(kSaveDir + "loss.tsv");
  ofs << "time\tepoch\tstep\trecon_loss\tkl_loss" << std::endl;
  ofs << std::fixed;
  std::cout << std::fixed;

  Timer timer;
  timer.start();

  torch::optim::Adam optimizer(vae->parameters(), torch::optim::AdamOptions(1e-3));

  constexpr int64_t kEpochs = 2000;
  constexpr int64_t kBatchSize = 512;
  std::mt19937 engine(std::random_device{}());
  const int64_t data_num = data_vector.size();
  torch::Device device(torch::kCUDA);
  vae->to(device);

  for (int64_t epoch = 1; epoch <= kEpochs; epoch++) {
    std::shuffle(data_vector.begin(), data_vector.end(), engine);
    for (int64_t i = 0; i < data_num; i += kBatchSize) {
      const int64_t step = i / kBatchSize + 1;
      std::vector<torch::Tensor> batch(
        data_vector.begin() + i, data_vector.begin() + std::min(i + kBatchSize, data_num));

      torch::Tensor x = torch::stack(batch, 0).to(device);
      auto [mean, var] = vae->encode(x);
      torch::Tensor z = vae->sample(mean, var);
      torch::Tensor y = vae->decode(z);
      torch::Tensor kl_loss = -0.5 * (1 + torch::log(var) - mean.pow(2) - var).sum(1).mean(0);
      torch::Tensor recon_loss =
        torch::binary_cross_entropy(y, x, {}, torch::Reduction::None).mean(1).sum({1, 2}).mean(0);
      torch::Tensor loss = kl_loss + recon_loss;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      std::stringstream ss;
      ss << timer.elapsed_time() << "\t" << epoch << "\t" << step << "\t"
         << recon_loss.item<float>() << "\t" << kl_loss.item<float>() << std::endl;
      std::cout << ss.str();
      ofs << ss.str();

      if (epoch % 100 == 0 && step == 1) {
        const std::string epoch_str =
          (std::stringstream() << std::setfill('0') << std::setw(4) << epoch).str();
        save_image(x[0], kSaveDir + "/images/input_" + epoch_str + ".png");
        save_image(y[0], kSaveDir + "/images/predict_" + epoch_str + ".png");
      }
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
