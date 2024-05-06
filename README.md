## Introduction

Disordered networks of $\mathrm{Au}$-nanoparticles have the potential to exploit physical processes for computations.
The nanoparticles of size $\sim 20$ nm serve as islands (or neurons) on which electrons can settle. As the nanoparticles are connected by organic molecules (1-octanethiols, which act as synapses), electrons can tunnel between neighbouring islands if differences in free energy are sufficient. Applying external voltages via Ti/Au control-electrodes enables to modify the internal dynamics. The electrode voltages act like genes and can be tuned by genetic algorithms or backpropagation. Subsequently, configurations are sought in which the system can perform a given task while being fast, energy-efficient and reconfigurable.

Generally, rate equations can be solved via Kinetic Monte Carlo (KMC) algorithms which is an inherently stochastic approach. If one tries to formulate an exact deterministic solution of the time evolution of the master equation, one faces the challenge of exponential computational complexity due to the exponentially large phase space. Thus, exact solutions are restricted to very small systems. One way, to formulate a deterministic solution with only polynomial complexity is the use of the mean-field approximation. For the current system of electron dynamics in systems of connected gold nanoparticles a first-order mean-field approximation has been introduced in . These ideas have been also used to solve the master equation describing triplet-triplet annihilation in phosphorescent emission layers of  light-emitting diodes.  

Here we provide a systematic approach which generalizes the previous meanfield approximation to higher orders. In practice, we will explore the second order method and show that it clearly outperforms the first order method albeit being still very efficient. The solution of the master equation is implemented in python and explored for different applications. In particular, an illustrative step towards the analysis of time-dependency is taken by analysing nonlinear input/output-relationships.

## Theory

The network is modelled as a collection of $N$ gold-nanoparticles arranged in a grid lattice, some of which are connected to electrodes. An occupation number $n \in \mathbb{Z}$ is assigned to each particle (island) representing the number of excess electrons on that island. This description will suffice to build the phase space $\Omega$. For $N$ particles, the phase space is thus


$$
    \Omega = \mathbb{Z}^N
$$

The state of the system is described by a state vector $\vec{n} \in \Omega$ which contains occupation numbers for excess electrons on each island. The network topology is encoded in a capacity matrix, linking nearest neighbours by their mutual capacity. The internal electrostatic energy can thus be calculated. Due to their proximity, electrons can tunnel through junctions between nearest neighbours. The tunnel rates will be described according to a zero-dimensional \emph{orthodox tunnel theory}. The tunnel rate $\Gamma$ for an electron from one position to another depends on the difference in Helmholtz free energy $\Delta F$ associated with this tunnel process

$$
    \Gamma = - \frac{\Delta F}{e^2 R} \cdot \left[ 1 - \mathrm{exp\,\frac{\Delta F}{k_B T}} \right]^{-1}
$$


We restrict ourselves to single-electron-tunneling.

It is important to mention that the free energy $\Delta F$, governing the underlying transition rate, not only depends on the population of the two involved islands but rather on the complete state vector $\vec{n}$ due to the network of capacities.  

A distribution function $\rho$ over phase space $\Omega$ is introduced. It assigns probability $\rho(\vec{n})$ to each possible system state $\vec{n}$ \cite{MasterEq}\cite{Jonas}\cite{Lawrence}. The tunnel rates describe the tunnelling of electrons between islands or electrodes and can be interpreted as the transition rate from states $\vec{n}$ to $\vec{m}$, denoted as $\Gamma_{\vec{n}\,\vec{m}}$. Hence, the rate for an electron to tunnel from island $i$ to island $j$ is the transition rate from state $\vec{n}$ to state $\vec{n} - \vec{e_i} + \vec{e_j}$, where the latter is missing one electron at index $i$ and having an additional one at index $j$. $\vec{e_i}$ is the i-th basis vector in $\Omega$. If an electrode is connected to island $i$, two more transition rates are possible: For an electron tunnelling towards the island, a transition from state $\vec{n}$ to state $\vec{n} + \vec{e_i}$, the rate $\Gamma_{\vec{n}\,\vec{n} + \vec{e_i}}$ is associated. For the reverse process, a minus sign must be taken. All other transition rates are zero.

The dynamics of the distribution function are governed by the master equation

$$
    \partial_t\, \rho(\vec{n}) = \sum_{\vec{m} \neq \vec{n}}( \Gamma_{\vec{m}\,\vec{n}} \,\rho (\vec{m}) - \Gamma_{\vec{n}\,\vec{m}} \,\rho (\vec{n}))
$$

The equilibrium distribution function is found when $\partial_t \, \rho(\vec{n}) = 0 \;\forall \vec{n}$. Afterwards, expectation values of quantities of interest can be taken. Most importantly, the currents flowing to the system via the electrodes.


## See Full Details here: https://arxiv.org/abs/2402.12223
