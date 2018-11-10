functions {
  vector log_segf(vector f,      // frequency
                  int    N,      // data size
                  real   fci,    // Large event corner frequency
                  real   fc1,    // Small event corner frequency
                  real   ni,     // Large event dropoff
                  real   n1,     // Small event dropoff
	          real   gi,     //
	          real   g1,     //
	          real   Mr) {   // Magnitude ratio

    vector[N] ly;	

    for (n in 1:N) {
      ly[n] = log(Mr)
            + log(1+(f[n]/fci)^(ni*gi))/gi
	    - log(1+(f[n]/fc1)^(n1*g1))/g1;
    }

    return ly;
  }
}

data {
  int<lower=0> N;        // Number measured frequencies
  int<lower=0> J;        // Number of ratios
  vector[N]  f;        // Measured frequencies
  vector[N] ly;
//vector[N] li[J];
  real mu_Mr[J]; // Magnitude ratio prior
}

parameters {
  real<lower=0> sig_ef;
  real<lower=0> sig_en;
  real<lower=0> sig;

 // real<lower=0> eta_f[L];
 // real<lower=0> eta_n[L];

  real<lower=1> fc1[J];
  real<lower=1> fci[J];
  real<lower=1> n1[J];
  real<lower=1> ni[J];
  real<lower=1> Mr[J];
}

model {
  // Scale priors
  sig_ef ~ cauchy( 0.0, 2.5 ) T[0.0,]; // Truncated Cauchy
  sig_en ~ cauchy( 0.0, 2.5 ) T[0.0,];
  sig  ~ cauchy( 0.0, 2.5 ) T[0.0,];

  // Azimuth bin effects
 // eta_f[L] ~ lognormal( 0.0, sig_ef );
 // eta_n[L] ~ lognormal( 0.0, sig_en );

 for (j in 1:J) {
    // Large event priors
    fci[j] ~ lognormal(-2.8, 0.7); // Large event corner frequency
    ni[j]  ~ lognormal( 0.7, 0.4); // Large event dropoff

    //
    fc1[j] ~ lognormal( 0.0, 1.5 );
    n1[j]  ~ lognormal( 0.7, 0.4 ); 
    Mr[j]  ~ lognormal( mu_Mr[j], 0.4 );

    // EGF model
    target += normal_lpdf(ly[j] | log_segf( f, N, fci[j], fc1[j],
                                           ni[j], n1[j], 2.0, 2.0, Mr[j]),
                                  sig);
  }

}
