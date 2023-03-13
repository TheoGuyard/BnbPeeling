using BnbPeeling
using Distributions
using LinearAlgebra
using MathOptInterface
using Random
using Test

const MOI = MathOptInterface

function synthetic_data(
    k::Int,
    m::Int,
    n::Int,
    σ::Float64,
    ρ::Float64,
    τ::Float64,
)

    x = zeros(n)
    s = Int.(floor.(collect(1:n/k:n)))
    a = rand(Normal(0., σ), k)
    x[s] = (σ == 0.) ? ones(k) : a + sign.(a)

    μ = zeros(n)
    N = [collect(1:n);]
    Σ = ρ .^ (abs.(repeat(N, inner=(1, n)) - repeat(N', inner=(n, 1))))
    L = MvNormal(μ, Σ)
    A = zeros(m, n)
    for j in 1:m
        A[j, :] = rand(L)
    end

    w = A * x
    ϵ = randn(m)
    ϵ *= ((τ != Inf) ? norm(w, 2) / (sqrt(τ) * norm(ϵ, 2)) : 0.0)
    y = w + ϵ

    return x, A, y
end

@testset "solve_bnb" begin
    x, A, y = synthetic_data(5, 100, 300, 0.1, 0.1, 100.)
    Mval    = 1.5 * norm(x, Inf)
    λmax    = Mval * norm(A' * y, Inf)
    result  = BnbPeeling.solve_bnb(A, y, 0.1 * λmax, Mval, verbosity=true)
    @test result.termination_status == MOI.OPTIMAL
end
