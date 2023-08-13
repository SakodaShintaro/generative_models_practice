#include "vae.hpp"

VAEImpl::VAEImpl(int64_t z_dim)
{
  enc1_ = register_module("enc1_", torch::nn::Linear(28 * 28, 200));
  enc2_ = register_module("enc2_", torch::nn::Linear(200, 200));
  enc_mean_ = register_module("enc_mean_", torch::nn::Linear(200, z_dim));
  enc_var_ = register_module("enc_var_", torch::nn::Linear(200, z_dim));
  dec1_ = register_module("dec1_", torch::nn::Linear(z_dim, 200));
  dec2_ = register_module("dec2_", torch::nn::Linear(200, 200));
  dec3_ = register_module("dec3_", torch::nn::Linear(200, 28 * 28));
}

std::pair<torch::Tensor, torch::Tensor> VAEImpl::encode(torch::Tensor x)
{
  x = enc1_(x);
  x = torch::relu(x);
  x = enc2_(x);
  x = torch::relu(x);
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
  x = dec1_(x);
  x = torch::relu(x);
  x = dec2_(x);
  x = torch::relu(x);
  x = dec3_(x);
  x = torch::sigmoid(x);
  return x;
}

torch::Tensor VAEImpl::forward(torch::Tensor x)
{
  auto [mean, var] = encode(x);
  torch::Tensor z = sample(mean, var);
  torch::Tensor y = decode(z);
  return y;
}
