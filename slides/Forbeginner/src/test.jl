#%%
using MLDatasets
using Images
using JuMP, Clp

#%%
dataset = MNIST(:train)
id_one = 7 # findfirst(dataset.targets .== 1)
id_two = 26 # findfirst(dataset.targets .== 2)

#%%
one = Matrix(dataset.features[:, :, id_one]')
two = Matrix(dataset.features[:, :, id_two]')

one_vec = vec(one)
two_vec = vec(two)

#%%
colorview(Gray, one)

#%%
colorview(Gray, two) .* two_amount

#%%
dim = 28
siz = dim * dim
C = zeros(Float64, siz, siz)
for i in 1:siz
    xᵢ, yᵢ = ((i - 1) % dim) + 1, cld(i, dim)
    for j in 1:siz
        xⱼ, yⱼ = ((j - 1) % dim) + 1, cld(j, dim)

        C[i, j] = (xᵢ - xⱼ)^2 + (yᵢ - yⱼ)^2
    end
end

#%%
function gen_quant(α, β, C, λ)
    amount_α = sum(α)
    amount_β = sum(β)
    α = α ./ amount_α
    β = β ./ amount_β

    model = Model(Clp.Optimizer)
    @variable(model, P[1:siz, 1:siz] .>= 0)
    @variable(model, Q[1:siz, 1:siz] .>= 0)
    @objective(model, Min, (1 - λ) * sum(C[i, j] * P[i, j] for i in 1:siz for j in 1:siz) + λ * sum(C[i, j] * Q[i, j] for i in 1:siz for j in 1:siz))
    @constraint(model, P * ones(siz) .== α)
    @constraint(model, Q' * ones(siz) .== β)
    @constraint(model, P' * ones(siz) .== Q * ones(siz))
    @constraint(model, sum(Q) == 1.0)
    optimize!(model)

    if termination_status(model) != OPTIMAL
        throw(ErrorException("wrong"))
    end

    γ = value.(Q) * ones(siz)
    γ = γ .* ((1 - λ) * amount_α + λ * amount_β)

    reshape(γ, dim, dim)
end

#%%
n = 50
λs = range(0, 1, length=n)
mids = zeros(Float64, n, dim, dim)

for i in 1:n
    λ = λs[i]
    mids[i, :, :] .= gen_quant(one_vec, two_vec, C, λ)
end

#%% 一度にすべての分布を求める
function gen_seq(α, β, C, n)
    amount_α = sum(α)
    amount_β = sum(β)
    α = α ./ amount_α
    β = β ./ amount_β

    model = Model(Clp.Optimizer)
    @variable(model, P[1:siz, 1:siz, 1:n] .>= 0)
    @objective(model, Min,
        sum(dot(C, P[:, :, k]) for k in 1:n)
    )
    @constraint(model, P[:, :, 1] * ones(siz) .== α)
    @constraint(model, P[:, :, n]' * ones(siz) .== β)
    for k in 1:n-1
        @constraint(model, P[:, :, k]' * ones(siz) .== P[:, :, k+1] * ones(siz))
    end
    for k in 1:n
        @constraint(model, sum(P[:, :, k]) == 1.0)
    end

    optimize!(model)
    if termination_status(model) != OPTIMAL
        throw(ErrorException("wrong"))
    end

    value.(P)
end

#%%
ret = gen_seq(one_vec, two_vec, C, 30)

#%%
