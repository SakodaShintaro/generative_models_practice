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
  torch::nn::ModuleList enc_conv_list_ = nullptr;
  torch::nn::Linear enc_mean_ = nullptr;
  torch::nn::Linear enc_var_ = nullptr;
  torch::nn::Linear dec_linear_ = nullptr;
  torch::nn::ModuleList dec_conv_list_ = nullptr;

  int64_t mid_h_;
  int64_t mid_w_;
};
TORCH_MODULE(VAE);

#endif  // VAE_HPP
