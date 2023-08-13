#ifndef VAE_HPP
#define VAE_HPP

#include <torch/torch.h>

class VAEImpl : public torch::nn::Module
{
public:
  VAEImpl(int64_t z_dim);
  std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);
  static torch::Tensor sample(torch::Tensor mean, torch::Tensor var);
  torch::Tensor decode(torch::Tensor x);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Linear enc1_ = nullptr;
  torch::nn::Linear enc2_ = nullptr;
  torch::nn::Linear enc_mean_ = nullptr;
  torch::nn::Linear enc_var_ = nullptr;
  torch::nn::Linear dec1_ = nullptr;
  torch::nn::Linear dec2_ = nullptr;
  torch::nn::Linear dec3_ = nullptr;
};
TORCH_MODULE(VAE);

#endif  // VAE_HPP
