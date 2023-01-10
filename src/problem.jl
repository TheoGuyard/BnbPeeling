struct Problem
    A::Matrix
    y::Vector
    λ::Float64
    Mval::Float64
    m::Int
    n::Int
    function Problem(
        A::Matrix,
        y::Vector,
        λ::Float64,
        Mval::Float64,
    )   
        m, n = size(A)
        @assert length(y) == m
        @assert λ > 0.
        @assert Mval > 0.
        @assert all(norm(ai) ≈ 1. for ai in eachcol(A))
        return new(A, y, λ, Mval, m, n)
    end
end

function objective(prob::Problem, x::Vector{Float64}, w::Vector{Float64})
    all(-prob.Mval .<= x .<= prob.Mval) || return Inf
    u = prob.y - w
    f = 0.5 * (u' * u)
    g = norm(x, 0.)
    return f + prob.λ * g
end
function objective(prob::Problem, x::Vector{Float64})
    all(-prob.Mval .<= x .<= prob.Mval) || return Inf
    w = prob.A * x
    return objective(prob, x, w)
end