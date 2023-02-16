using ArgParse
using BnbPeeling
using BnbScreening
using CPLEX
using Distributions
using JuMP
using LinearAlgebra
using PyCall
using Random
using RCall

CALIBRATE_R = joinpath(@__DIR__, "calibrate.R")

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
    for ai in eachcol(A)
        ai ./= norm(ai)
    end

    w = A * x
    ϵ = randn(m)
    ϵ *= ((τ != Inf) ? norm(w, 2) / (sqrt(τ) * norm(ϵ, 2)) : 0.0)
    y = w + ϵ

    return x, A, y
end

function calibrate(x::Vector, A::Matrix, y::Vector)
    result = R"source($(CALIBRATE_R)); fit_path_l0learn($x, $A, $y)"
    result = rcopy(result)
    x0 = result[:x]
    λ = result[:l]
    return x0, λ
end

function solve_cplex(
    A::Matrix, 
    y::Vector, 
    λ::Float64, 
    M::Float64;
    maxtime::Float64=60.,
    )

    problem = Model(CPLEX.Optimizer)
    set_optimizer_attribute(problem, "CPX_PARAM_TILIM", maxtime)
    set_optimizer_attribute(problem, "CPX_PARAM_EPINT", 1e-8)
    set_optimizer_attribute(problem, "CPXPARAM_MIP_Tolerances_MIPGap", 1e-8)
    set_silent(problem)

    n = size(A, 2)
    Q = A' * A
    q = A' * y
    c = y' * y
    
    @variable(problem, x[1:n])
    @variable(problem, z[1:n], Bin)
    @objective(
        problem, 
        Min, 
        0.5 * x' * Q * x - q' * x + 0.5 * c + λ * sum(z)
    )
    @constraint(problem, -M.*z .<= x)
    @constraint(problem, x .<= M.*z)

    optimize!(problem)

    result = Dict(
        :termination_status => JuMP.termination_status(problem),
        :solve_time         => JuMP.solve_time(problem),
        :node_count         => JuMP.node_count(problem),
        :objective_value    => JuMP.objective_value(problem),
        :x                  => JuMP.value.(problem[:x]),
    )

    return result
end

function solve_l0bnb(
    A::Matrix, 
    y::Vector, 
    λ::Float64, 
    M::Float64;
    maxtime::Float64=60.,
    )

    L0BNB  = pyimport("l0bnb")    
    tree   = L0BNB.BNBTree(A, y, int_tol=1e-8, rel_tol=1e-8)
    solve  = tree.solve(λ, 0., M, time_limit=maxtime, verbose=false)
    prob   = Problem(A, y, λ, M)
    result = Dict(
        :termination_status => solve[3] < maxtime ? MOI.OPTIMAL : MOI.TIME_LIMIT,
        :solve_time         => solve[3],
        :node_count         => tree.number_of_nodes,
        :objective_value    => objective(prob, solve[2]),
        :x                  => solve[2],
    )

    return result
end

function solve_sbnb(
    A::Matrix, 
    y::Vector, 
    λ::Float64, 
    M::Float64;
    maxtime::Float64=60.,
    )

    params = BnbScreening.BnbParams(
        maxtime     = maxtime,
        verbosity   = false,
        screening   = false,
        tol         = 1e-8,
    )
    result = BnbScreening.solve_bnb(A, y, λ, M, bnbparams=params)
    result = Dict(
        :termination_status => result.termination_status,
        :solve_time         => result.solve_time,
        :node_count         => result.node_count,
        :objective_value    => result.objective_value,
        :x                  => result.x,
    )

    return result
end

function solve_sbnbn(
    A::Matrix, 
    y::Vector, 
    λ::Float64, 
    M::Float64;
    maxtime::Float64=60.,
    )

    params = BnbScreening.BnbParams(
        maxtime     = maxtime,
        verbosity   = false,
        screening   = true,
        tol         = 1e-8,
    )
    result = BnbScreening.solve_bnb(A, y, λ, M, bnbparams=params)
    result = Dict(
        :termination_status => result.termination_status,
        :solve_time         => result.solve_time,
        :node_count         => result.node_count,
        :objective_value    => result.objective_value,
        :x                  => result.x,
    )

    return result
end