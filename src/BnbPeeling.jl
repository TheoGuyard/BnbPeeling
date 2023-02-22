module BnbPeeling

using Dates
using Distributions
using LinearAlgebra
using MathOptInterface
using Printf
using Random

const MOI = MathOptInterface

version() = "v0.1"
authors() = "Theo Guyard"
contact() = "theo.guyard@insa-rennes.fr"
license() = "AGPL 3.0"

include("problem.jl")
include("solver.jl")
include("bounding.jl")
include("accelerations.jl")

export Problem, objective
export AbstractBoundingSolver, BoundingType
export BnbParams, BnbResults, solve_bnb

end
