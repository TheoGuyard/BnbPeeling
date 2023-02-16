function l0screening!(
    A::Matrix,
    y::Vector,
    λ::Float64,
    Mpos::Vector,
    Mneg::Vector,
    x::Vector, 
    w::Vector, 
    u::Vector, 
    q::Vector, 
    ub::Float64,
    dv::Float64, 
    S0::BitArray,
    S1::BitArray,
    Sb::BitArray,
    Sbi::BitArray, 
    Sbb::BitArray, 
    S1i::BitArray,
    )
    for i in findall(Sbi .| Sbb)
        if dv + λ * max(-q[i], 0.0) > ub
            # Move i from Sbi or Sbb to S0
            Sbi[i] = false
            Sbb[i] = false
            Sb[i] = false
            S0[i] = true
            if x[i] != 0.0
                axpy!(-x[i], A[:, i], w)
                copy!(u, y - w)
                x[i] = 0.0
            end
            Mpos[i] = 0.
            Mneg[i] = 0.
        elseif dv + λ * max(q[i], 0.0) > ub
            # Move i from Sbi or Sbb to S1i
            Sbi[i] = false
            Sbb[i] = false
            S1i[i] = true
            Sb[i] = false
            S1[i] = true
        end
    end
    return nothing
end

function l1screening!(
    A::Matrix,
    y::Vector,
    λ::Float64,
    Mpos::Vector,
    Mneg::Vector,
    x::Vector,
    w::Vector,
    u::Vector,
    v::Vector,
    gap::Float64,
    Sb0::BitArray, 
    Sbi::BitArray, 
    Sbb::BitArray, 
    )

    radius = sqrt(2.0 * gap)
    for i in findall(Sbi)
        vi = v[i]
        if (abs(vi) + radius < λ / Mpos[i]) & (abs(vi) + radius < -λ / Mneg[i])
            # Move i from Sbi to Sb0
            if x[i] != 0.
                axpy!(-x[i], A[:, i], w)  
                copy!(u, y - w)
                x[i] = 0.
            end
            Sbi[i] = false
            Sb0[i] = true
        elseif (abs(vi) - radius > λ / Mpos[i]) & (abs(vi) - radius > -λ / Mneg[i])        
            # Move i from Sbi to Sbb
            Sbi[i] = false
            Sbb[i] = true
        end
    end

    return nothing
end

function bigmpeeling!(
    A::Matrix,
    y::Vector,
    λ::Float64,
    Mpos::Vector,
    Mneg::Vector,
    x::Vector, 
    w::Vector, 
    u::Vector, 
    v::Vector, 
    q::Vector, 
    ub::Float64,
    dv::Float64, 
    S0::BitArray,
    Sb::BitArray,
    Sbi::BitArray, 
    Sbb::BitArray,
    )

    flag_update = false

    for i in findall(Sbi .| Sbb)
        vi = v[i]
        α = (dv + λ * max(-q[i], 0.) - ub) / vi
        if vi < 0.
            # Peel Mpos
            αpos = max(Mneg[i] + α, 0.)
            if αpos < Mpos[i]
                # (αpos == 0.) && println("pos $i : $(Mpos[i]) -> $αpos")
                if x[i] > αpos
                    axpy!(αpos - x[i], A[:, i], w)
                    x[i] = αpos
                    flag_update = true
                end
                Mpos[i] = αpos
            end
        elseif vi > 0.
            # Peel Mneg
            αneg = min(Mpos[i] + α, 0.)
            if αneg > Mneg[i]
                # (αneg == 0.) && println("neg $i : $(Mneg[i]) -> $αneg")
                if x[i] < αneg
                    axpy!(αneg - x[i], A[:, i], w)
                    x[i] = αneg
                    flag_update = true
                end
                Mneg[i] = αneg
            end
        end
    end

    flag_update && copy!(u, y - w)

    return nothing
end