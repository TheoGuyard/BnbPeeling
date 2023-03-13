"""
    Problem

L0-penalized least-squares problem with Big-M constraint.
"""
struct Problem
    A::Matrix
    y::Vector
    λ::Float64
    Mval::Float64
    m::Int
    n::Int
    a::Vector
    function Problem(A::Matrix, y::Vector, λ::Float64, Mval::Float64)
        m, n = size(A)
        a = [norm(ai, 2)^2 for ai in eachcol(A)]
        @assert length(y) == m
        @assert λ > 0.0
        @assert Mval > 0.0
        return new(A, y, λ, Mval, m, n, a)
    end
end

"""
    objective(prob::Problem, x::Vector, Ax::Vector)

Evalutes the objective of a [`Problem`](@ref) at `x` when the value of `Ax` is
already known.
"""
function objective(prob::Problem, x::Vector, Ax::Vector)
    all(-prob.Mval .<= x .<= prob.Mval) || return Inf
    u = prob.y - Ax
    f = 0.5 * (u' * u)
    g = norm(x, 0.0)
    return f + prob.λ * g
end

"""
    objective(prob::Problem, x::Vector)

Evalutes the objective of a [`Problem`](@ref) at `x`.
"""
function objective(prob::Problem, x::Vector)
    all(-prob.Mval .<= x .<= prob.Mval) || return Inf
    w = prob.A * x
    return objective(prob, x, w)
end

function Base.show(io::IO, problem::Problem)
    println(io, "L0-penalized problem")
    println(io, "  Dims    : $(problem.m) x $(problem.n)")
    println(io, "  Mval    : $(problem.Mval)")
    println(io, "  λ       : $(round(problem.λ, digits=4))")
    print(
        io,
        "  λ/λmax  : $(round(problem.λ / (problem.Mval * norm(problem.A' * problem.y, Inf)), digits=4))",
    )
end
