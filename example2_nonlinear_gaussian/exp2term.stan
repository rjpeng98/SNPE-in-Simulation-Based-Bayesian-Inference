data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real<lower=0, upper=10> theta1;
  real<lower=0, upper=10> theta2;
  real<lower=0, upper=10> theta3;
  real<lower=0, upper=10> theta4;
  real<lower=0> sigma2;  // variance parameter
}

transformed parameters {
  vector[N] mu;
  real<lower=0> sigma;  // standard deviation
  
  for (i in 1:N) {
    mu[i] = theta1 * exp(-theta2 * x[i])
          + theta3 * exp(-(theta3 + theta4) * x[i]);
  }
  
  sigma = sqrt(sigma2);  // convert variance to std dev
}

model {
  // Priors
  // theta1, theta2, theta3, theta4 ~ Uniform(0, 10) [implicit via bounds]
  
  // sigma2 ~ InvGamma(2, 1)
  sigma2 ~ inv_gamma(2, 1);
  
  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  for (i in 1:N) {
    y_rep[i] = normal_rng(mu[i], sigma);
  }
}