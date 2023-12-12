#include "vae.hpp"

VAEImpl::VAEImpl(int64_t z_dim)
{
  using namespace torch::nn;

  const int64_t init_h = 96;
  const int64_t init_w = 96;

  const int64_t base_ch = 16;
  std::vector<int64_t> ch_list = {
    3, base_ch * 1, base_ch * 2, base_ch * 4, base_ch * 8,
  };
  const int64_t n_layers = ch_list.size() - 1;

  enc_conv_list_ = torch::nn::ModuleList();
  for (int64_t i = 0; i < n_layers; i++) {
    enc_conv_list_->push_back(
      Conv2d(Conv2dOptions(ch_list[i], ch_list[i + 1], 3).stride(2).padding(1)));
  }
  register_module("enc_conv_list_", enc_conv_list_);

  mid_h_ = init_h / (1ull << n_layers);
  mid_w_ = init_w / (1ull << n_layers);

  const int64_t out_channels = ch_list.back() * mid_h_ * mid_w_;
  enc_mean_ = register_module("enc_mean_", torch::nn::Linear(out_channels, z_dim));
  enc_var_ = register_module("enc_var_", torch::nn::Linear(out_channels, z_dim));

  dec_linear_ = register_module("dec_linear_", torch::nn::Linear(z_dim, out_channels));

  dec_conv_list_ = torch::nn::ModuleList();
  for (int64_t i = n_layers - 1; i >= 0; i--) {
    dec_conv_list_->push_back(
      ConvTranspose2d(ConvTranspose2dOptions(ch_list[i + 1], ch_list[i], 2).stride(2)));
  }
  register_module("dec_conv_list_", dec_conv_list_);
}

std::pair<torch::Tensor, torch::Tensor> VAEImpl::encode(torch::Tensor x)
{
  for (const auto & layer : *enc_conv_list_) {
    x = layer->as<torch::nn::Conv2d>()->forward(x);
    x = torch::relu(x);
  }
  x = x.view({x.size(0), -1});

  torch::Tensor mean = enc_mean_(x);
  torch::Tensor var = torch::softplus(enc_var_(x));
  return {mean, var};
}

torch::Tensor VAEImpl::sample(torch::Tensor mean, torch::Tensor var)
{
  torch::Tensor epsilon = torch::randn(mean.sizes()).to(mean.device());
  return mean + torch::sqrt(var) * epsilon;
}

torch::Tensor VAEImpl::decode(torch::Tensor x)
{
  x = dec_linear_(x);
  x = x.view({x.size(0), -1, mid_h_, mid_w_});

  const int64_t n_layers = dec_conv_list_->size();
  for (int64_t i = 0; i < n_layers; i++) {
    const auto & layer = (*dec_conv_list_)[i];
    x = layer->as<torch::nn::ConvTranspose2d>()->forward(x);
    x = (i == n_layers - 1 ? torch::sigmoid(x) : torch::relu(x));
  }
  return x;
}

torch::Tensor VAEImpl::forward(torch::Tensor x)
{
  auto [mean, var] = encode(x);
  torch::Tensor z = sample(mean, var);
  torch::Tensor y = decode(z);
  return y;
}
