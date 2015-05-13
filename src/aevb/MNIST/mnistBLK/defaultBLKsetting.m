function [blk, settings]=defaultBLKsetting(K)
    % sample initial latent variables {G,C,K} provided the hyperblketers

    settings.iterations=100;
%     settings.burnin=0;
%     settings.thinout=1;
%     settings.m_aux=3;

    settings.sample_alpha = 0;
    settings.sample_sigma_x=0;
    settings.sample_sigma_g=0;
    settings.sample_sigma_noise=0;

    % {alpha}, 
    % {mu_x, sigma_x}           : noise on X
    % {mu_g, sigma_g}           : noise on G
    % {mu_noise, sigma_noise}   : noise on Y
    blk.alpha=1;
    blk.mu_x=0;
    blk.sigma_x=1;
    blk.mu_g=0;
    blk.sigma_g=1;
    blk.mu_noise=0;
    blk.sigma_noise=0.1;
    
    blk.noise_hyper_a=1;
    blk.noise_hyper_b=.1;
    blk.sigma_g_hyper_a=1;
    blk.sigma_g_hyper_b=1;
    
    blk.K = K;
end