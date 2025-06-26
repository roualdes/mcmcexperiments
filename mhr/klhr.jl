using BridgeStan
const BS = BridgeStan

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim

function bsmodel_ld(bsmodel)
    function bsm_ld(x)
        ld = try
            BS.log_density(bsmodel, x)
        catch e
            -Inf
        end
        return ld
    end
    return bsm_ld
end

function bsmodel_ldg(bsmodel)
    function bsm_ldg(x)
        ld, g = try
            BS.log_density_gradient(bsmodel, x)
        catch e
            D = BS.param_unc_num(bsmodel)
            -Inf, zeros(D)
        end
        return ld, g
    end
    return bsm_ldg
end

function LVI(ldg, eta, x, w, rho, origin)
    N = length(x)
    dm = zero(eltype(x))
    ds = zero(eltype(x))
    out = zero(eltype(x))
    for n in 1:N
        z = exp(eta[2]) * x[n] + eta[1]
        xi = rho * z + origin
        ld, g = ldg(xi)
        out += w[n] * ld
        dm += w[n] * (g' * rho)
        ds += w[n] * (g' * rho * exp(eta[2]) * x[n])
    end
    return -out - eta[2], -dm, -ds - 1
end

function dadvi_esque(ldg, rho, origin; N = 20, tol = 1e-2)
    x, w = gausshermite(N, normalize = true);

    function fg!(F, G, eta)
        out, dm, ds = LVI(ldg, eta, x, w, rho, origin)
        if G !== nothing
            G[1] = dm
            G[2] = ds
        end
        if F !== nothing
            return out
        end
    end

    init = zeros(2)
    obj = OnceDifferentiable(Optim.only_fg!(fg!), init)
    method = BFGS()
    opts = Optim.Options(x_abstol = tol, x_reltol = tol, f_abstol = tol, f_reltol = tol, g_abstol = tol)
    state = Optim.initial_state(method, opts, obj, init)
    r = Optim.optimize(obj, init, method, opts, state)
    mkl = r.minimizer[1]
    skl = sqrt(state.invH[1, 1])
    return mkl, skl
end

function overrelaxed_proposal(NDist, K)
    u = cdf(NDist, 0)
    r = rand(Binomial(K, u))
    up = if r > K - r
        v = rand(Beta(K - r + 1, 2r - K))
        u * v
    elseif r < K - r
        v = rand(Beta(r + 1, K - 2r))
        1 - (1 - u) * v
    elseif r == K - r
        u
    end
    return quantile(NDist, up)
end

function klhr(bsmodel; M = 1_000, N = 10, overrelaxed = false, K = 32, tol = 1e-2, init = [])
    D = BS.param_unc_num(bsmodel)
    draws = zeros(M, D)
    draws[1, :] = if length(init) == D
        draws[1, :] .= init
    else
        rand(Uniform(-1, 1), D)
    end

    mvn_direction = MvNormal(zeros(D), ones(D))
    acceptance_rate = 0.0

    bsm_ld = bsmodel_ld(bsmodel)
    bsm_ldg = bsmodel_ldg(bsmodel)

    for m in 2:M
        rho = rand(mvn_direction)
        rho ./= norm(rho)

        prev = draws[m - 1, :]
        mkl, skl = dadvi_esque(bsm_ldg, rho, prev; tol)

        N = Normal(mkl, skl)
        z = if overrelaxed
            overrelaxed_proposal(N, K)
        else
            rand(N)
        end

        prop = rho * z + prev

        a = bsm_ld(prop)
        a -= bsm_ld(prev)
        a += logpdf(N, 0)
        a -= logpdf(N, z)

        accept = log(rand()) < min(0, a)
        draws[m, :] = accept * prop + (1 - accept) * prev
        acceptance_rate += (accept - acceptance_rate) / (m - 1)
    end

    println("acceptance rate = $(acceptance_rate)")
    return draws
end
