#=================#
# Import packages #
#=================#
using Parameters
using JuMP
import Ipopt
using ProgressMeter
using PrettyTables
using GLMakie
using CairoMakie

#==============#
# Housekeeping #
#==============#
PWD = pwd()
VER = "V9"
if Sys.iswindows()
    FL = "\\"
else
    FL = "/"
end
PATH = mkpath(PWD * FL * VER)
PATH_FIG = mkpath(PATH * FL * "Figures")

#==============#
# BP functions #
#==============#
τ_1(x_1, x_2, μ) = x_1 * μ + (1.0 - x_2) * (1.0 - μ)
μ_1(x_1, x_2, μ) = x_1 * μ / τ_1(x_1, x_2, μ)
τ_2(x_1, x_2, μ) = (1.0 - x_1) * μ + x_2 * (1.0 - μ)
μ_2(x_1, x_2, μ) = (1.0 - x_1) * μ / τ_2(x_1, x_2, μ)
H(μ) = -(μ * log(μ) + (1.0 - μ) * log((1.0 - μ)))
c(x_1, x_2, μ) = (1.0 / log(2.0)) * (H(μ) - τ_1(x_1, x_2, μ) * H(μ_1(x_1, x_2, μ)) - τ_2(x_1, x_2, μ) * H(μ_2(x_1, x_2, μ)))

#=====================#
# inflation functions #
#=====================#
x_r(ω_i, x_T, ν_1, ν_2) = ω_i == 1 ? x_T + ν_1 : x_T - ν_2
x_e(μ, x_T, ν_1, ν_2) = μ * x_r(1, x_T, ν_1, ν_2) + (1.0 - μ) * x_r(2, x_T, ν_1, ν_2) # x_T + ν * (2.0 * μ - 1.0) if symmetric ν

#========================#
# CB objective functions #
#========================#
obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2) = δ * c(x_1, x_2, μ_0) * (μ_0_c * (ω_1 + γ * (x_e(μ_0, x_T, ν_1, ν_2) - x_r(1, x_T, ν_1, ν_2)))^2.0 + (1.0 - μ_0_c) * (ω_2 + γ * (x_e(μ_0, x_T, ν_1, ν_2) - x_r(2, x_T, ν_1, ν_2)))^2.0)
obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2) = (1.0 - δ * c(x_1, x_2, μ_0)) * μ_0_c * (x_1 * (ω_1 + γ * (x_e(μ_1(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(1, x_T, ν_1, ν_2)))^2.0 + (1.0 - x_1) * (ω_1 + γ * (x_e(μ_2(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(1, x_T, ν_1, ν_2)))^2.0)
obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2) = (1.0 - δ * c(x_1, x_2, μ_0)) * (1.0 - μ_0_c) * ((1.0 - x_2) * (ω_2 + γ * (x_e(μ_1(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(2, x_T, ν_1, ν_2)))^2.0 + x_2 * (ω_2 + γ * (x_e(μ_2(x_1, x_2, μ_0), x_T, ν_1, ν_2) - x_r(2, x_T, ν_1, ν_2)))^2.0)
obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α) = α * (μ_0_c * (x_r(1, x_T, ν_1, ν_2) - x_T)^2 + (1.0 - μ_0_c) * (x_r(2, x_T, ν_1, ν_2) - x_T)^2)
obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α) = obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2) + obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2) + obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2) + obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α)
# obj_CB(x_1, x_2, para::Dict{String,Any}) =
    # obj_CB_μ_0(x_1, x_2, para["μ_0"], para["μ_0_c"], para["ω_1"], para["ω_2"], para["δ"], para["γ"], para["x_T"], para["ν_1"], para["ν_2"]) +
    # obj_CB_1(x_1, x_2, para["μ_0"], para["μ_0_c"], para["ω_1"], para["δ"], para["γ"], para["x_T"], para["ν_1"], para["ν_2"]) +
    # obj_CB_2(x_1, x_2, para["μ_0"], para["μ_0_c"], para["ω_2"], para["δ"], para["γ"], para["x_T"], para["ν_1"], para["ν_2"]) +
    # obj_CB_x(para["μ_0_c"], para["x_T"], para["ν_1"], para["ν_2"], para["α"])
# obj_CB_para_list = ["μ_0", "μ_0_c", "ω_1", "ω_2", "δ", "γ", "x_T", "ν_1", "ν_2", "α"]

#======================#
# benchmark parameters #
#======================#
@with_kw struct Benchmark_Parameters
    δ::Float64 = 0.5
    ω_1::Float64 = 1.0
    ω_2::Float64 = -1.0
    μ_0::Float64 = 0.5
    μ_0_diff::Float64 = 0.0
    μ_0_c::Float64 = 0.5 # μ_0 * (1.0 + μ_0_diff / 100)
    γ::Float64 = 10.0
    x_T::Float64 = 2.0
    ν_1::Float64 = 1.0
    ν_2::Float64 = 1.0
    α::Float64 = 1.0
    ϵ_x::Float64 = 1E-6
    ϵ_tol::Float64 = 1E-8
end
BP = Benchmark_Parameters()
PATH_FIG_γ = mkpath(PATH_FIG * FL * "γ_$(floor(Int, BP.γ))")

#==================#
# benchmark result #
#==================#
function optimal_x_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, ν_1, ν_2, α::Float64, ϵ_x::Float64, ϵ_tol::Float64)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    return objective_value(model), value(x_1), value(x_2)
end
println("Find the minimum of $(optimal_x_func(BP.μ_0,BP.μ_0_c,BP.ω_1,BP.ω_2,BP.δ,BP.γ,BP.x_T,BP.ν_1,BP.ν_2,BP.α, BP.ϵ_x, BP.ϵ_tol)[1]) at (x_1,x_2) = ($(optimal_x_func(BP.μ_0,BP.μ_0_c,BP.ω_1,BP.ω_2,BP.δ,BP.γ,BP.x_T,BP.ν_1,BP.ν_2,BP.α, BP.ϵ_x, BP.ϵ_tol)[2]), $(optimal_x_func(BP.μ_0,BP.μ_0_c,BP.ω_1,BP.ω_2,BP.δ,BP.γ,BP.x_T,BP.ν_1,BP.ν_2,BP.α, BP.ϵ_x, BP.ϵ_tol)[3])")

#===========================================#
# handy functions for optimal flexibility ν #
#===========================================#
function optimal_flexibility_func!(BP::Benchmark_Parameters, res::Array{Float64,4},
    TA::String, TA_size::Int64, TA_grid::Vector{Float64},
    ν_1_size::Int64, ν_1_grid::Vector{Float64}, ν_2_size::Int64, ν_2_grid::Vector{Float64})

    # unpack benchmark parameters
    @unpack μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, ϵ_x, ϵ_tol = BP

    # create the iterator indices for multi-threading
    ind_nested_loop = collect(Iterators.product(1:ν_2_size, 1:ν_1_size, 1:TA_size))

    # timer
    pp = Progress(ν_2_size * ν_1_size * TA_size)
    update!(pp, 0)
    jj = Threads.Atomic{Int}(0)
    ll = Threads.SpinLock()

    # start looping
    Threads.@threads for (ν_2_i, ν_1_i, TA_i) in ind_nested_loop

        # create the parameter dictionary
        obj_CB_para = Dict([("μ_0", μ_0), ("μ_0_c", μ_0_c), ("ω_1", ω_1), ("ω_2", ω_2), ("δ", δ), ("γ", γ), ("x_T", x_T), ("ν_1", ν_1), ("ν_2", ν_2), ("α", α), ("ϵ_x", ϵ_x), ("ϵ_tol", ϵ_tol)])
        obj_CB_para[TA] = TA_grid[TA_i]
        obj_CB_para["ν_1"] = ν_1_grid[ν_1_i]
        obj_CB_para["ν_2"] = ν_2_grid[ν_2_i]

        # slove CB's optimization problem for (x_1, x_2)
        obj_opt, x_1_opt, x_2_opt = optimal_x_func(obj_CB_para["μ_0"], obj_CB_para["μ_0_c"], obj_CB_para["ω_1"], obj_CB_para["ω_2"], obj_CB_para["δ"], obj_CB_para["γ"], obj_CB_para["x_T"], obj_CB_para["ν_1"], obj_CB_para["ν_2"], obj_CB_para["α"], obj_CB_para["ϵ_x"], obj_CB_para["ϵ_tol"])

        # save results
        @inbounds res[TA_i, ν_1_i, ν_2_i, 1] = ν_1_grid[ν_1_i]
        @inbounds res[TA_i, ν_1_i, ν_2_i, 2] = ν_2_grid[ν_2_i]
        @inbounds res[TA_i, ν_1_i, ν_2_i, 3] = obj_opt
        @inbounds res[TA_i, ν_1_i, ν_2_i, 4] = x_1_opt
        @inbounds res[TA_i, ν_1_i, ν_2_i, 5] = x_2_opt
        @inbounds res[TA_i, ν_1_i, ν_2_i, 6] = μ_1(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, ν_1_i, ν_2_i, 7] = μ_2(x_1_opt, x_2_opt, obj_CB_para["μ_0"])
        @inbounds res[TA_i, ν_1_i, ν_2_i, 8] = 1.0 - obj_CB_para["δ"] * c(x_1_opt, x_2_opt, obj_CB_para["μ_0"])

        # update timer
        Threads.atomic_add!(jj, 1)
        Threads.lock(ll)
        update!(pp, jj[])
        Threads.unlock(ll)
    end
    return nothing
end

# function ν_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, ν_1::Float64, ν_2::Float64, α::Float64, ϵ_x::Float64, ϵ_tol::Float64)

#     # unpack benchmark parameters
#     @unpack μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, α, ϵ_x, ϵ_tol = BP

#     # create the parameter dictionary
#     obj_CB_para = Dict([("μ_0", μ_0), ("μ_0_c", μ_0_c), ("ω_1", ω_1), ("ω_2", ω_2), ("δ", δ), ("γ", γ), ("x_T", x_T), ("ν_1", ν_1), ("ν_2", ν_2), ("α", α), ("ϵ_x", ϵ_x), ("ϵ_tol", ϵ_tol)])
#     obj_CB_para[TA] = TA_
#     obj_CB_para["ν_1"] = ν_1
#     obj_CB_para["ν_2"] = ν_2

#     # slove CB's optimization problem for (x_1, x_2) and report optimized minimum value
#     return optimal_x_func(obj_CB_para["μ_0"], obj_CB_para["μ_0_c"], obj_CB_para["ω_1"], obj_CB_para["ω_2"], obj_CB_para["δ"], obj_CB_para["γ"], obj_CB_para["x_T"], obj_CB_para["ν_1"], obj_CB_para["ν_2"], obj_CB_para["α"], obj_CB_para["ϵ_x"], obj_CB_para["ϵ_tol"])[1]

#     optimal_x_func(μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, ϵ_x, ϵ_tol)[1]

# end

function optimal_ν_func(μ_0::Float64, μ_0_c::Float64, ω_1::Float64, ω_2::Float64, δ::Float64, γ::Float64, x_T::Float64, α::Float64, ϵ_x::Float64, ϵ_tol::Float64)

    # slove CB's optimization flexibility problem for (ν_1, ν_2)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, 0.0 <= ν_1, start = 0.0)
    @variable(model, 0.0 <= ν_2, start = 0.0)
    _obj_ν(ν_1, ν_2) = optimal_x_func(μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2, α, ϵ_x, ϵ_tol)[1]
    @objective(model, Min, _obj_ν(ν_1, ν_2))
    optimize!(model)

    # report optimized minimum value
    return objective_value(model), value(ν_1), value(ν_2)
end

# function optimal_flexibility_func!(BP::Benchmark_Parameters, res::Array{Float64,4},
#     TA::String, TA_size::Int64, TA_grid::Vector{Float64},
#     ν_1_size::Int64, ν_1_grid::Vector{Float64}, ν_2_size::Int64, ν_2_grid::Vector{Float64})

#     # unpack benchmark parameters
#     @unpack μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, α, ϵ_x, ϵ_tol = BP

#     # timer
#     pp = Progress(ν_2_size * ν_1_size * TA_size)
#     update!(pp, 0)
#     jj = Threads.Atomic{Int}(0)
#     ll = Threads.SpinLock()

#     # start looping
#     Threads.@threads for TA_i in TA_size

#         # extract TA value
#         TA_ = TA_grid[TA_i]

#         # slove CB's optimization problem for (ν_1, ν_2)
#         model = Model(Ipopt.Optimizer)
#         set_silent(model)
#         set_attribute(model, "tol", ϵ_tol)
#         @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
#         @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
#         _obj_flexibility(ν_1, ν_2) = flexibility_func(BP, TA, TA_, ν_1, ν_2)
#         @objective(model, Min, _obj_flexibility(ν_1, ν_2))
#         optimize!(model)

#         # create the parameter dictionary
#         obj_CB_para = Dict([("μ_0", μ_0), ("μ_0_c", μ_0_c), ("ω_1", ω_1), ("ω_2", ω_2), ("δ", δ), ("γ", γ), ("x_T", x_T), ("ν_1", 0.0), ("ν_2", 0.0), ("α", α)])
#         obj_CB_para[TA] = TA_
#         obj_CB_para["ν_1"] = value(ν_1)
#         obj_CB_para["ν_2"] = value(ν_2)

#         # save results
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 1] = ν_1_grid[ν_1_i]
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 2] = ν_2_grid[ν_2_i]
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 3] = objective_value(model)
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 4] = value(x_1)
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 5] = value(x_2)
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 6] = μ_1(value(x_1), value(x_2), obj_CB_para["μ_0"])
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 7] = μ_2(value(x_1), value(x_2), obj_CB_para["μ_0"])
#         @inbounds res[TA_i, ν_1_i, ν_2_i, 8] = 1.0 - obj_CB_para["δ"] * c(value(x_1), value(x_2), obj_CB_para["μ_0"])

#         # update timer
#         Threads.atomic_add!(jj, 1)
#         Threads.lock(ll)
#         update!(pp, jj[])
#         Threads.unlock(ll)
#     end
#     return nothing
# end

#==============================#
# benchmark result - ν and μ_0 #
#==============================#
μ_0_grid = collect(0.05:0.05:0.95)
μ_0_size = length(μ_0_grid)
ν_1_grid = collect(0.05:0.001:0.11)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.05:0.001:0.11)
ν_2_size = length(ν_2_grid)
res = zeros(μ_0_size, ν_1_size, ν_2_size, 8)
optimal_flexibility_func!(BP, res, "μ_0", μ_0_size, μ_0_grid, ν_1_size, ν_1_grid, ν_2_size, ν_2_grid)

# rounding numbers
# res = round.(res, digits=4)

# minimizer and minimum
res_obj_min_ind = argmin(res[:, :, :, 3], dims=(2, 3))
ν_1_min = [res[μ_0_i, res_obj_min_ind[μ_0_i][2], res_obj_min_ind[μ_0_i][3], 1] for μ_0_i = 1:μ_0_size]
ν_2_min = [res[μ_0_i, res_obj_min_ind[μ_0_i][2], res_obj_min_ind[μ_0_i][3], 2] for μ_0_i = 1:μ_0_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\mu_0$", ylabel=L"$\nu$")
scatterlines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5, markersize=20)
scatterlines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dot, linewidth=5, markersize=20, marker=:xcross)
# lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5)
# lines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=5)
axislegend(position=:rt, nbanks=1, patchsize=(70, 30))
fig

# save figures
filename = "fig_optimal_ν_μ_0" * ".pdf"
save(PATH_FIG_γ * FL * filename, fig)
filename = "fig_optimal_ν_μ_0" * ".png"
save(PATH_FIG_γ * FL * filename, fig)

# minimizer and minimum
x_1_min = [res[μ_0_i, res_obj_min_ind[μ_0_i][2], res_obj_min_ind[μ_0_i][3], 4] for μ_0_i = 1:μ_0_size]
x_2_min = [res[μ_0_i, res_obj_min_ind[μ_0_i][2], res_obj_min_ind[μ_0_i][3], 5] for μ_0_i = 1:μ_0_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\mu_0$")
scatterlines!(ax, μ_0_grid, x_1_min, label=L"$x_1$", color=:blue, linestyle=nothing, linewidth=5, markersize=20)
scatterlines!(ax, μ_0_grid, x_2_min, label=L"$x_2$", color=:red, linestyle=:dot, linewidth=5, markersize=20, marker=:xcross)
# lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5)
# lines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=5)
axislegend(position=:ct, nbanks=1, patchsize=(70, 30))
fig

# save figures
filename = "fig_optimal_ν_μ_0_x" * ".pdf"
save(PATH_FIG_γ * FL * filename, fig)
filename = "fig_optimal_ν_μ_0_x" * ".png"
save(PATH_FIG_γ * FL * filename, fig)

#================================#
# benchmark result - ν and μ_0_c #
#================================#
μ_0_c_grid = collect(0.05:0.05:0.95)
μ_0_c_size = length(μ_0_c_grid)
ν_1_grid = collect(0.05:0.025:1.65)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.05:0.025:1.65)
ν_2_size = length(ν_2_grid)
res = zeros(μ_0_c_size, ν_1_size, ν_2_size, 8)
optimal_flexibility_func!(BP, res, "μ_0_c", μ_0_c_size, μ_0_c_grid, ν_1_size, ν_1_grid, ν_2_size, ν_2_grid)

# minimizer and minimum
res_obj_min_ind = argmin(res[:, :, :, 3], dims=(2, 3))
ν_1_min = [res[μ_0_c_i, res_obj_min_ind[μ_0_c_i][2], res_obj_min_ind[μ_0_c_i][3], 1] for μ_0_c_i = 1:μ_0_c_size]
ν_2_min = [res[μ_0_c_i, res_obj_min_ind[μ_0_c_i][2], res_obj_min_ind[μ_0_c_i][3], 2] for μ_0_c_i = 1:μ_0_c_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\mu_0^c$", ylabel=L"$\nu$")
scatterlines!(ax, μ_0_c_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5, markersize=20)
scatterlines!(ax, μ_0_c_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dot, linewidth=5, markersize=20, marker=:xcross)
# lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5)
# lines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=5)
axislegend(position=:ct, nbanks=1, patchsize=(70, 30))
fig

# save figures
filename = "fig_optimal_ν_μ_0_c" * ".pdf"
save(PATH_FIG_γ * FL * filename, fig)
filename = "fig_optimal_ν_μ_0_c" * ".png"
save(PATH_FIG_γ * FL * filename, fig)

#==============================#
# benchmark result - ν and ω_1 #
#==============================#
ω_1_grid = collect(0.5:0.05:1.5)
ω_1_size = length(ω_1_grid)
ν_1_grid = collect(0.35:0.005:0.65)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.35:0.005:0.65)
ν_2_size = length(ν_2_grid)
res = zeros(ω_1_size, ν_1_size, ν_2_size, 8)
optimal_flexibility_func!(BP, res, "ω_1", ω_1_size, ω_1_grid, ν_1_size, ν_1_grid, ν_2_size, ν_2_grid)

# minimizer and minimum
res_obj_min_ind = argmin(res[:, :, :, 3], dims=(2, 3))
ν_1_min = [res[ω_1_i, res_obj_min_ind[ω_1_i][2], res_obj_min_ind[ω_1_i][3], 1] for ω_1_i = 1:ω_1_size]
ν_2_min = [res[ω_1_i, res_obj_min_ind[ω_1_i][2], res_obj_min_ind[ω_1_i][3], 2] for ω_1_i = 1:ω_1_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\omega_1$", ylabel=L"$\nu$")
scatterlines!(ax, ω_1_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5, markersize=20)
scatterlines!(ax, ω_1_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dot, linewidth=5, markersize=20, marker=:xcross)
# lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5)
# lines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=5)
axislegend(position=:rb, nbanks=1, patchsize=(70, 30))
fig

# save figures
filename = "fig_optimal_ν_ω_1" * ".pdf"
save(PATH_FIG_γ * FL * filename, fig)
filename = "fig_optimal_ν_ω_1" * ".png"
save(PATH_FIG_γ * FL * filename, fig)

#==============================#
# benchmark result - ν and ω_2 #
#==============================#
ω_2_grid = collect(-1.5:0.05:-0.5)
ω_2_size = length(ω_2_grid)
ν_1_grid = collect(0.35:0.005:0.65)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.35:0.005:0.65)
ν_2_size = length(ν_2_grid)
res = zeros(ω_2_size, ν_1_size, ν_2_size, 8)
optimal_flexibility_func!(BP, res, "ω_2", ω_2_size, ω_2_grid, ν_1_size, ν_1_grid, ν_2_size, ν_2_grid)

# minimizer and minimum
res_obj_min_ind = argmin(res[:, :, :, 3], dims=(2, 3))
ν_1_min = [res[ω_2_i, res_obj_min_ind[ω_2_i][2], res_obj_min_ind[ω_2_i][3], 1] for ω_2_i = 1:ω_2_size]
ν_2_min = [res[ω_2_i, res_obj_min_ind[ω_2_i][2], res_obj_min_ind[ω_2_i][3], 2] for ω_2_i = 1:ω_2_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\omega_2$", ylabel=L"$\nu$")
scatterlines!(ax, ω_2_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5, markersize=20)
scatterlines!(ax, ω_2_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dot, linewidth=5, markersize=20, marker=:xcross)
# lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=5)
# lines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=5)
axislegend(position=:rt, nbanks=1, patchsize=(70, 30))
fig

# save figures
filename = "fig_optimal_ν_ω_2" * ".pdf"
save(PATH_FIG_γ * FL * filename, fig)
filename = "fig_optimal_ν_ω_2" * ".png"
save(PATH_FIG_γ * FL * filename, fig)