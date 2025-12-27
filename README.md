# Autoencoder & Variational Autoencoder Implementation
Implementation of Autoencoders and Variational Autoencoders using PyTorch on the MNIST dataset.


### Autoencoder
Autoencoders learn compressed representations by encoding input data into a lower-dimensional latent space and then reconstructing it back to the original space.

**Loss Function**: Mean Squared Error (MSE)

### Variational Autoencoder
VAEs are generative models that learn a probabilistic mapping to the latent space, enabling sampling and generation of new data.

**Loss Function**: Reconstruction Loss + KL Divergence

