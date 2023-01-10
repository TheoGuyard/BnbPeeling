# module BnbPeeling

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

abstract type AbstractSolver end
@enum BoundingType begin
    LOWER
    UPPER
end

include("problem.jl")
include("data.jl")
include("bnb.jl")
include("cd.jl")
include("accelerations.jl")

# end