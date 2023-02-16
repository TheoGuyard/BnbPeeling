# Safe peeling for L0-regularized least-squares

This repository contains numerical procedures linked to the following paper.

> The citation will be available soon

If you encounter a bug or something unexpected, please let me know by raising an [issue](https://github.com/TheoGuyard/BnbPeeling.jl/issues).

## Requirements

`BnbPeeling.jl` is tested against `Julia v1.8`. Please refer to Julia's [download page](https://julialang.org/downloads/) for install instructions.
Parts of our code require the use of [CPLEX](https://www.ibm.com/fr-fr/analytics/cplex-optimizer) solver through [CPLEX.jl](https://github.com/jump-dev/CPLEX.jl). You cannot use CPLEX without having purchased and installed a copy of CPLEX Optimization Studio from [IBM](https://www.ibm.com). However, CPLEX is available for free to [academics and students](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students). We recommend you to follow carefully the install instructions described at [CPLEX.jl](https://github.com/jump-dev/CPLEX.jl) to install the package.
Our package also depends on [L0Learn](https://github.com/hazimehh/L0Learn) and [L0bnb](https://github.com/alisaab/l0bnb) that are respectively called through [RCall.jl](https://github.com/JuliaInterop/RCall.jl) and [PyCall.jl](https://github.com/JuliaPy/PyCall.jl).
Please, refer to these packages for install instructions.

## Quick start

1. Clone the repository
```bash
$ git clone https://github.com/TheoGuyard/BnbPeeling.jl
```
2. Enter in the project root folder
```bash
$ cd BnbPeeling.jl
```
3. Install the package using [Julia's Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/) in REPL mode
```julia
pkg> activate .
pkg> instantiate
```
4. You can check that everything works well by running
```julia
pkg> test
```

## EUSIPCO experiments

Experiments presented in the paper submitted to [EUSIPCO 2023](http://eusipco2023.org) are located in the `exp/` folder. The script corresponding to the Figure 1 of the paper is `exp/eusipco.jl`.
You can run experiments directly from the project root folder as follows.
```bash
$ julia --project=. -t 1 exp/eusipco.jl <k> <m> <n> <σ> <ρ> <τ> <γ> <solver> <maxtime> --seed <seed>
```

The `--project=.` flag activates the dependencies. The `-t 1` flag restricts computations to only 1 CPU in order to avoid bias due to parallelization capabilities.
This script constructs a synthetic instance of the problem and runs a solver with some specified time limit.
A seed can be specified for reproducibility.
You can refer to the paper linked to this package for more details.

**Parameters specifications:**
* `k::Int` : sparsity level, must be positive and lower than `n`
* `m::Int` : number of rows in the feature matrix, must be positive
* `n::Int` : number of columns in the feature matrix, must be positive
* `σ::Float64` : non-zero entries amplitude parameter, must be positive
* `ρ::Float64` : correlation parameter, must be between `0` and `1`
* `τ::Float64` : signal-to-noise ration, must be positive
* `γ::Float64` : big-m calibration factor, must be larger than `1`
* `solver::String` : solver to run, choices are `cplex`, `l0bnb`, `sbnb`, `sbnbn`, `sbnbp`
* `maxtime::Float64` : maximum time allowed in second, must be positive
* `--seed::Int` : seed to use, optional, must be strictly positive

**Example:**
```bash
$ julia --project=. -t 1 exp/eusipco.jl 5 100 200 0.1 0.1 10.0 2.0 sbnbp 60.0 --seed 42
```

## Licence

This software is distributed under the [GNU AGPL-3](https://www.gnu.org/licenses/agpl-3.0.en.html) license.

## Cite this work

If you use this package for your own work, please consider citing it as :

> The citation will be available soon