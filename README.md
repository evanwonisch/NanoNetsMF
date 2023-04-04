# NanoNetsMF
A mean-field implementation to solve for charge configurations in gold-nanoparticle networks


We have seen that:

## Implementation of Lawrence Meanfield (LMF)

1. the single-electron-transistor of lawrence and KMC is very accurate

I have implemented my own solution to the master equation and LMF and KMC lie in very good agreement for set.

2. Due to theoretical advancements, I could see that LMF approximation lies in a simplified dummy-distribution to estimate expecation values. This approximation uses a probability distriubtion which, given its mean, assigns probability to the floor and ceiling value of the mean.

3. It can be seen, that the output error current is positively correlated with error between LMF approximate distribution and the exact master equation distribution.

## Second Order Meanfield (MF2)

1. By a second order method, we could improve the match between MF-approximate distribution and real distribution to get better expectation values. Notably for expect(I**2) and so on.

2. It is seen, that the correct distribution obtained by master equation statisfies the equilibrium properties of the first two moments.

3. The second order Meanfield for the single-electron-transistor is implemented and with it the discrete gaussian and Lawrence' distribution. Convergence is stable. Now need for evaluation of MF2 wrt. Master Equation, LMF.

4. The 2nd order method is a great improvement

## Implementation of Master Equation

1. It is seen, that the correct distribution can have a big variance, which must be accounted for in MF.


## Comparison of KMC and LMF for larger systems

1. LMF produces very accurate means wich deviate just abt. 0.1 electron charges per island. Still, the output currents are not living up to that. We might find a solution by a better MF-distribution approximation.

2. It seems that the output-currents from MF deviate systematically from the KMC data. This might hint to another problem. An empirical factor of 1 + e (electron charge) drastically imporves the accuracy for large systems.
