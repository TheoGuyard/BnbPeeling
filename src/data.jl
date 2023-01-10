function synthetic_data(k, m, n, ρ, s)
    
    x = zeros(Float64, n)
    S = Int.(floor.(collect(1:n/k:n)))
    a = rand(Normal(0., 1.), k)
    x[S] = a + sign.(a)

    μ = zeros(n)
    N = [collect(1:n); ]
    Σ = ρ .^ (abs.(repeat(N, inner=(1, n)) - repeat(N', inner=(n, 1))))
    L = MvNormal(μ, Σ)
    A = zeros(m, n)
    for j in 1:m
        A[j, :] = rand(L)
    end

    for ai in eachcol(A)
        normalize!(ai)
    end

    w = A * x
    σ = std(w) / s
    ϵ = (s != Inf) ? rand(Normal(0., σ), m) : zeros(m)
    y = w + ϵ

    return x, A, y
end


