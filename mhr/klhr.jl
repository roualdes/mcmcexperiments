using BridgeStan
const BS = BridgeStan

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim


function LVI(bsmodel, eta, x, w, rho, origin)
    N = length(x)
    dm = zero(eltype(x))
    ds = zero(eltype(x))
    out = zero(eltype(x))
    for n in 1:N
        z = exp(eta[2]) * x[n] + eta[1]
        xi = rho * z + origin
        ld, g = BS.log_density_gradient(bsmodel, xi)
        out += w[n] * ld
        dm += w[n] * (g' * rho)
        ds += w[n] * (g' * rho * exp(eta[2]) * x[n])
    end
    return -out - eta[2], -dm, -ds - 1
end

function dadvi_esque(bsmodel, rho, origin; N = 20)
    x, w = gausshermite(N, normalize = true);

    function fg!(F, G, eta)
        out, dm, ds = LVI(bsmodel, eta, x, w, rho, origin)
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
    opts = Optim.Options()
    state = Optim.initial_state(method, opts, obj, init)
    r = Optim.optimize(obj, init, method, opts, state)
    mkl = r.minimizer[1]
    skl = sqrt(state.invH[1, 1])
    return mkl, skl
end

function klhr(bsmodel; M = 1_000, N = 10)
    D = BS.param_unc_num(bsmodel)
    draws = zeros(M, D)
    draws[1, :] = rand(Uniform(-1, 1), D)
    mvn_direction = MvNormal(zeros(D), ones(D))
    acceptance_rate = 0.0

    for m in 2:M
        rho = rand(mvn_direction)
        rho ./= norm(rho)

        prev = draws[m - 1, :]
        mkl, skl = dadvi_esque(bsmodel, rho, prev)

        N = Normal(mkl, skl)
        z = rand(N)
        prop = rho * z + prev

        a = BS.log_density(bsmodel, prop)
        a -= BS.log_density(bsmodel, prev)
        a += logpdf(N, 0)
        a -= logpdf(N, z)

        accept = log(rand()) < min(0, a)
        draws[m, :] = accept * prop + (1 - accept) * prev
        acceptance_rate += (accept - acceptance_rate) / (m - 1)
    end

    println("acceptance rate = $(acceptance_rate)")
    return draws
end
