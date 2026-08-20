data {
  int<lower=1> N;           // number of observations
  vector[N] x;              // design points (X values)
  vector[N] y;              // observed data (Y values)
  real<lower=0> nu;         // degrees of freedom for Student-t
}

parameters {
  real<lower=0, upper=10> theta1;
  real<lower=0, upper=10> theta2;
  real<lower=0, upper=10> theta3;
  real<lower=0, upper=10> theta4;
  real<lower=0> sigma2;     // variance parameter
}

transformed parameters {
  real<lower=0> sigma = sqrt(sigma2);
  vector[N] mu;
  
  // Model: Y(X) = θ₁*exp(-θ₂*X) + θ₃*exp(-(θ₃+θ₄)*X)
  for (i in 1:N) {
    mu[i] = theta1 * exp(-theta2 * x[i]) + theta3 * exp(-(theta3 + theta4) * x[i]);
  }
}

model {
  // Priors
  theta1 ~ uniform(0, 10);
  theta2 ~ uniform(0, 10);
  theta3 ~ uniform(0, 10);
  theta4 ~ uniform(0, 10);
  sigma2 ~ inv_gamma(2, 1);
  
  // Likelihood with Student-t errors
  y ~ student_t(nu, mu, sigma);
}