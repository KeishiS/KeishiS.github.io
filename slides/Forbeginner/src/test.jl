#%%
using MLDatasets
using Images
using JuMP, Clp

#%%
dataset = MNIST(:train)
id_one = 7 # findfirst(dataset.targets .== 1)
id_two = 26 # findfirst(dataset.targets .== 2)
id_three = findfirst(dataset.targets .== 3)

#%%
one = Matrix(dataset.features[:, :, id_one]')
two = Matrix(dataset.features[:, :, id_two]')
three = Matrix(dataset.features[:, :, id_three]')

one_vec = vec(one)
two_vec = vec(two)
three_vec = vec(three)

#%%
colorview(Gray, one)

#%%
colorview(Gray, two)

#%%
colorview(Gray, three)

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
mids = zeros(Float64, dim, dim, n)

#%%
for i in 1:n
    λ = λs[i]
    mids[:, :, i] .= gen_quant(one_vec, two_vec, C, λ)
end

#%%
function gen_quant2(α, β, C, λ; ϵ=1, n_loop=50)
    K = exp.(-C ./ ϵ)
    u = ones(Float64, siz, 2)
    v = zeros(Float64, siz, 2)
    for _ in 1:n_loop
        v[:, 1] = α ./ (K' * u[:, 1])
        v[:, 2] = β ./ (K' * u[:, 2])

        tmp = ((K * v[:, 1]) .^ (1 - λ)) .* ((K * v[:, 2]) .^ λ)
        u[:, 1] = tmp ./ (K * v[:, 1])
        u[:, 2] = tmp ./ (K * v[:, 2])
    end

    ((K * v[:, 1]) .^ (1 - λ)) .* ((K * v[:, 2]) .^ λ)
end

#%%
mids = zeros(Float64, dim, dim, n)
for i in 1:n
    λ = λs[i]
    ret = gen_quant2(two_vec, three_vec, C, λ; ϵ=1)
    mids[:, :, i] = reshape(ret, dim, dim)
    mids[:, :, i] = mids[:, :, i] ./ maximum(mids[:, :, i])
    save("./out/mid_$(lpad(i, 2, '0')).png", colorview(Gray, mids[:, :, i]))
end
