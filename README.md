# NanoNetsMF
A mean-field implementation to solve for charge configurations in gold-nanoparticle networks


We have seen that:

## Implementation of Lawrence Meanfield (LMF)

1. the single-electron-transistor of lawrence is much more accurate than expected due to systematic deviations between Jonas and my Codebase or different simulation parameters.

I have implemented my own solution to the master equation and LMF lies in very good agreement.

2. Due to theoretical advancements, I could see that LMF approximation lies in a simplified dummy-distribution to estimate expecation values. This approximation uses a probability distriubtion which, given its mean, assigns probability to the floor and ceiling value of the mean.

It can be seen, that the output error current is positively correlated with error between LMF approximate distribution and the exact master equation distribution.

3. Thus, by a second order method, we could improve the match between MF-approximate distribution and real distribution to get better expectation values.

## Implementation of Master Equation

## Comparison of KMC and LMF for larger systems

1. LMF produces very accurate means wich deviate just abt. 0.1 electron charges per island. Still, the output are not living up to that. We might find a solution by a better MF-distribution approximation.
