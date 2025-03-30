import numpy as np
import torch

# implements the Diffusion Denoising Process Sampler

class DDPMSampler:
    def __init__(self, generator, num_training_step = 1000, 
                 beta_start = 0.00085, beta_end = 0.0120):  # hyperparameters taken from original implementation
        
        self.generator = generator
        self.num_training_steps = num_training_step
        # the Gaussian distribution parameter at each timestep required for adding noise incrementally
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_step, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.timesteps = torch.from_numpy(np.arange(0, self.num_training_steps)[::-1].copy())

    def set_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def set_strength(self, strength : float = 0.8):
        '''
        sets strength to the noise added to the image
        higher strength => more noise added to the input image and thus further away from the image
        lower strength => less noise and thus closer to the image
        '''
        # the amount of noise to be added by controlling the number of timesteps
        first_step = self.num_inference_steps - int(strength * self.num_inference_steps)
        self.start_step = first_step
        self.timesteps = self.timesteps[first_step:]
    # the noise is added depending on the timestep based on the paper Denoising Diffusion probabilistic models paper
    # in the forward process, we add Gaussian noise of mean and variance which is timestep based and in the reverse process, the model
    # tries to find the error added over the timesteps through the mean and variance given by e_theta
    def add_noise(self, latents : torch.FloatTensor, timesteps : torch.LongTensor):
        alphas_cumprod = self.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
        timesteps = timesteps.to(device=latents.device)

        sqrt_alphas_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alphas_prod = sqrt_alphas_prod.flatten()
        # reshaping sqrt_alphas_prod tensor to latents' shape
        num_missing_dims = len(latents.shape) - len(sqrt_alphas_prod.shape)
        new_shape = sqrt_alphas_prod.shape + (1,)*num_missing_dims
        sqrt_alphas_prod = sqrt_alphas_prod.view(new_shape)

        sqrt_one_minus_alphas_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alphas_prod = sqrt_one_minus_alphas_prod.flatten()
        num_missing_dims = len(latents.shape) - len(sqrt_one_minus_alphas_prod.shape)
        new_shape = sqrt_one_minus_alphas_prod.shape + (1,)*num_missing_dims
        sqrt_one_minus_alphas_prod = sqrt_one_minus_alphas_prod.view(new_shape)

        # sample q(xt|x0) from eq 4 of the DDPM paper
        # X ~ N(mu,sigma), hence X = mu + sigma * N(0,1)
        # where mu = sqrt_alphas_prod * latents and sigma = sqrt_one_minus_alphas_prod
        noise = torch.randn(latents.shape, generator =self.generator, device=latents.device, dtype=latents.dtype)
        noisy_samples = sqrt_alphas_prod * latents + sqrt_one_minus_alphas_prod * noise
        return noisy_samples
    
    def _get_variance(self, timestep):
        prev_t = timestep - self.num_training_steps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one

        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1-alpha_prod_t_prev) / (1-alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance



        # compute the variance for the current timestep


    # this function performs the reverse of the forward diffusion process and gets the latent after removal of noise given the timestep
    # model_output is the approximated error added over timesteps to the image to noisify it.  
    # predict x_t-1 | x_t
    def step(self, timestep, latents, model_output):
        t = timestep
        prev_t = t - self.num_training_steps // self.num_inference_steps

        # compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # we now have an estimate of the error epsilon added at time step t from t-1 from the model
        # we now get an estimate of original latent at time 0
        pred_original_sample = (latents - beta_prod_t**0.5 * model_output) / alpha_prod_t ** 0.5
        # get the forward process posterior mean using x0 and xt
        # the following coeffs are found in the eq 7 of DDPM paper
        pred_original_sample_coeff = alpha_prod_t_prev ** 0.5 * current_beta_t / (1-alpha_prod_t)
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        mean_estimate = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # computation of variance to sample latent space from previous time step
        # x_sample = mean_estimate + std * noise , noise~N(0,1)
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape,generator =self.generator,
                                device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise 

        pred_prev_sample = mean_estimate + variance
        return pred_prev_sample












        
