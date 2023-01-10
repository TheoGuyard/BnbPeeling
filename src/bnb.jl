Base.@kwdef struct Options
    lb_solver::AbstractSolver           = CoordinateDescent(tolgap=1e-4, maxiter=10_000)
    ub_solver::AbstractSolver           = CoordinateDescent(tolgap=1e-8, maxiter=10_000)
    maxtime::Float64                    = 60.
    maxnode::Int                        = typemax(Int)
    tolgap::Float64                     = 0.
    tolint::Float64                     = 0.
    dualpruning::Bool                   = false
    l0screening::Bool                   = false
    l1screening::Bool                   = false
    bigmpeeling::Bool                   = false
    verbosity::Bool                     = false
    showevery::Int                      = 1
    trace::Bool                         = true
end

mutable struct Node 
    parent::Union{Node,Nothing}
    depth::Int
    S0::BitArray
    S1::BitArray
    Sb::BitArray
    Mpos::Vector{Float64}
    Mneg::Vector{Float64}
    lb::Float64
    ub::Float64
    x::Vector{Float64}
    w::Vector{Float64}
    u::Vector{Float64}
    x_ub::Vector{Float64}
    function Node(prob::Problem)
        return new(
            nothing,
            0,
            falses(prob.n),
            falses(prob.n),
            trues(prob.n),
            fill(prob.Mval, prob.n),
            fill(-prob.Mval, prob.n),
            -Inf,
            Inf,
            zeros(prob.n),
            zeros(prob.m),
            copy(prob.y),
            zeros(prob.n),
        )
    end
    function Node(parent::Node, j::Int, jval::Int, prob::Problem)
        child = new(
            parent,
            parent.depth + 1,
            copy(parent.S0),
            copy(parent.S1),
            copy(parent.Sb),
            copy(parent.Mpos),
            copy(parent.Mneg),
            copy(parent.lb),
            copy(parent.ub),
            copy(parent.x),
            copy(parent.w),
            copy(parent.u),
            copy(parent.x_ub),
        )
        fixto!(child, j, jval, prob)
        return child
    end
end

Base.@kwdef mutable struct Trace 
    ub::Vector{Float64}             = Vector{Float64}()
    lb::Vector{Float64}             = Vector{Float64}()
    node_count::Vector{Int}         = Vector{Int}()
    timer::Vector{Float64}          = Vector{Float64}()
    card_Sb::Vector{Int}            = Vector{Int}()
    card_S1::Vector{Int}            = Vector{Int}()
    card_S0::Vector{Int}            = Vector{Int}()
    spread::Vector{Float64}         = Vector{Float64}()
end

mutable struct BnB 
    status::MOI.TerminationStatusCode
    ub::Float64
    lb::Float64
    x::Vector{Float64}
    queue::Vector{Node}
    node_count::Int
    start_time::Float64
    function BnB(prob::Problem, x0::Vector{Float64})
        return new(
            MOI.OPTIMIZE_NOT_CALLED,
            objective(prob, x0),
            -Inf,
            x0,
            [Node(prob)],
            0,
            Dates.time(),
        )
    end
end

struct Result 
    termination_status::MOI.TerminationStatusCode
    solve_time::Float64
    node_count::Int
    objective_value::Float64
    relative_gap::Float64
    x::Vector{Float64}
    trace::Trace
    function Result(bnb::BnB, trace::Trace)
        return new(
            bnb.status,
            elapsed_time(bnb),
            bnb.node_count,
            bnb.ub,
            gap(bnb),
            bnb.x,
            trace,
        )
    end
end

function display_head()
    println(repeat("-", 60))
    print("  Time")
    print("   Nodes")
    print("   Lower")
    print("   Upper")
    print("     Gap")
    print("   %S0")
    print("   %S1")
    print("   %Sb")
    print("\n")
    println(repeat("-", 60))
    return nothing
end

function display_trace(prob, bnb, node)
    @printf " %5.2f" elapsed_time(bnb)
    @printf " %7d" bnb.node_count
    @printf " %7.2f" bnb.lb
    @printf " %7.2f" bnb.ub
    @printf "  %5.2f%%" 100 * gap(bnb)
    @printf "  %3d%%" 100 * sum(node.S0) / length(node.S0)
    @printf "  %3d%%" 100 * sum(node.S1) / length(node.S1)
    @printf "  %3d%%" 100 * sum(node.Sb) / length(node.Sb)
    @printf "  %3d%%" 100 * trace.spread
    println()
end

function display_tail()
    println(repeat("-", 60))
end

depth(node::Node) = sum(node.S0 .| node.S1)
elapsed_time(t0::Float64) = Dates.time() - t0
elapsed_time(bnb::BnB) = Dates.time() - bnb.start_time
gap(bnb::BnB) = abs(bnb.ub - bnb.lb) / (bnb.ub)
is_terminated(bnb::BnB) = (bnb.status != MOI.OPTIMIZE_NOT_CALLED)

function update_status!(bnb::BnB, options::Options)
    if elapsed_time(bnb) >= options.maxtime
        bnb.status = MOI.TIME_LIMIT
    elseif bnb.node_count >= options.maxnode
        bnb.status = MOI.ITERATION_LIMIT
    elseif gap(bnb) <= options.tolgap
        bnb.status = MOI.OPTIMAL
    elseif isempty(bnb.queue)
        bnb.status = MOI.OPTIMAL
    end
    return (bnb.status != MOI.OPTIMIZE_NOT_CALLED)
end

function next_node!(bnb::BnB, options::Options)
    node = pop!(bnb.queue)
    bnb.node_count += 1
    return node
end

function prune!(prob::Problem, bnb::BnB, node::Node, options::Options)
    pruning_test = (node.lb > bnb.ub)
    perfect_test = (abs(node.ub - node.lb) <= options.tolgap) 
    prune = (pruning_test | perfect_test)
    return prune
end

function branch!(prob::Problem, bnb::BnB, node::Node, options::Options)
    !any(node.Sb) && return nothing
    jSb = argmax(abs.(node.x[node.Sb])) 
    j = (1:prob.n)[node.Sb][jSb]
    node_j0 = Node(node, j, 0, prob)
    node_j1 = Node(node, j, 1, prob)
    push!(bnb.queue, node_j0)
    push!(bnb.queue, node_j1)
    return nothing
end

function fixto!(node::Node, j::Int, jval::Int, prob::Problem)
    node.Sb[j] || error("Branching index $j is already fixed")
    node.Sb[j] = false
    if jval == 0
        node.S0[j] = true
        node.Mpos[j] = 0.
        node.Mneg[j] = 0.
        if node.x[j] != 0.0
            axpy!(-node.x[j], prob.A[:, j], node.w)
            copy!(node.u, prob.y - node.w)
            node.x[j] = 0.0
        end
    elseif jval == 1
        node.S1[j] = true
    end
    return nothing
end

function update_bounds!(bnb::BnB, node::Node, options::Options)
    if (node.ub ≈ bnb.ub) & (norm(node.x_ub, 0) < norm(bnb.x, 0))
        bnb.ub = copy(node.ub)
        bnb.x = copy(node.x_ub)
        filter!(queue_node -> queue_node.lb < bnb.ub, bnb.queue)
    elseif node.ub < bnb.ub
        bnb.ub = copy(node.ub)
        bnb.x = copy(node.x_ub)
        filter!(queue_node -> queue_node.lb < bnb.ub, bnb.queue)
    end
    if isempty(bnb.queue)
        bnb.lb = min(node.lb, bnb.ub)
    else
        bnb.lb = minimum([queue_node.lb for queue_node in bnb.queue])
    end
end

function update_trace!(problem::Problem, trace::Trace, bnb::BnB, node::Node, options::Options)
    push!(trace.ub, bnb.ub)
    push!(trace.lb, bnb.lb)
    push!(trace.node_count, bnb.node_count)
    push!(trace.timer, elapsed_time(bnb))
    push!(trace.card_Sb, sum(node.Sb))
    push!(trace.card_S0, sum(node.S0))
    push!(trace.card_S1, sum(node.S1))
    push!(trace.spread, sum(node.Mpos - node.Mneg) / (2 * problem.Mval * problem.n))
    return nothing
end

function Base.show(io::IO, result::Result)
    println(io, "BnB result")
    println(io, "  Status     : $(result.termination_status)")
    println(io, "  Objective  : $(round(result.objective_value, digits=5))")
    println(io, "  Last gap   : $(round(100 * result.relative_gap, digits=5))%")
    println(io, "  Solve time : $(round(result.solve_time, digits=5)) seconds")
    println(io, "  Node count : $(result.node_count)")
    println(io, "  Non-zeros  : $(norm(result.x, 0))")
    print(io, "  Inf-norm x : $(norm(result.x, Inf))")
    return nothing
end

function solve(
    A::Matrix{Float64},
    y::Vector{Float64},
    λ::Float64,
    Mval::Float64;
    x0::Union{Vector{Float64},Nothing}=nothing,
    kwargs...
    )

    prob    = Problem(A, y, λ, Mval)
    x0      = isa(x0, Nothing) ? zeros(prob.n) : x0
    @assert length(x0) == prob.n
    bnb     = BnB(prob, x0)
    trace   = Trace()
    options = Options(; kwargs...)

    options.verbosity && display_head()
    while true
        update_status!(bnb, options) 
        is_terminated(bnb) && break
        node = next_node!(bnb, options)
        bound!(options.lb_solver, prob, bnb, node, options, LOWER)
        if !(prune!(prob, bnb, node, options))
            bound!(options.ub_solver, prob, bnb, node, options, UPPER)
            branch!(prob, bnb, node, options)
        end
        update_bounds!(bnb, node, options)
        options.trace && update_trace!(prob, trace, bnb, node, options)
        options.verbosity && bnb.node_count % options.showevery == 0 && display_trace(prob, bnb, node)
    end
    options.verbosity && display_tail()

    return Result(bnb, trace)
end