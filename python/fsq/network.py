import flax.linen as nn
from fsq import FSQ


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=3, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME')(x)
        return x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=64, kernel_size=(
            4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(
            4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(
            4, 4), strides=(2, 2), padding='SAME')(x)
        return x


class VQVAE(nn.Module):
    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantizer = FSQ(levels=[3, 3, 3])

    def __call__(self, x):
        encoded = self.encoder(x)
        quantized = self.quantizer.quantize(encoded)
        decoded = self.decoder(quantized)
        return decoded
