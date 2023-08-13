#include "main_functions.hpp"
#include "vae.hpp"

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cout << "Usage: vae_cpp <mode>" << std::endl;
    return 1;
  }

  const std::string mode = argv[1];

  if (mode == "train") {
    if (argc != 3) {
      std::cout << "Usage: vae_cpp train <input_dir>" << std::endl;
      return 1;
    }
    const std::string input_dir = argv[2];
    train(input_dir);
  } else if (mode == "generate") {
    if (argc != 3) {
      std::cout << "Usage: vae_cpp train <output_dir>" << std::endl;
      return 1;
    }
    const std::string output_dir = argv[2];
    generate(output_dir);
  } else {
    std::cerr << "Unknown mode: " << mode << std::endl;
    std::exit(1);
  }
}
