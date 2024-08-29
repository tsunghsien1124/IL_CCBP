#=================#
# Import packages #
#=================#
using JuMP
import Ipopt
using PrettyTables
using GLMakie
using CairoMakie

#==============#
# Housekeeping #
#==============#
PWD = pwd()
VER = "V9"
PATH = mkpath(PWD * "\\" * VER)
PATH_FIG = mkpath(PATH * "\\Figures")

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
obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2) = obj_CB_μ_0(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2) + obj_CB_1(x_1, x_2, μ_0, μ_0_c, ω_1, δ, γ, x_T, ν_1, ν_2) + obj_CB_2(x_1, x_2, μ_0, μ_0_c, ω_2, δ, γ, x_T, ν_1, ν_2) + obj_CB_x(μ_0_c, x_T, ν_1, ν_2, α)

#======================#
# benchmark parameters #
#======================#
δ = 0.5
ω_1 = 1.0
ω_2 = -1.0
μ_0 = 0.5
μ_0_diff = 0.0
μ_0_c = 0.5 # μ_0 * (1.0 + μ_0_diff / 100)
γ = 10.0
x_T = 2.0
# ν_1 = 1.0
# ν_2 = ν_1
α = 1.0
ϵ_x = 1E-6
ϵ_tol = 1E-8

#=========================#
# creating result folders #
#=========================#
PATH_FIG_γ = mkpath(PATH_FIG * "\\" * "γ_$(floor(Int, γ))")

#==================#
# benchmark result #
#==================#
# model = Model(Ipopt.Optimizer)
# set_silent(model)
# set_attribute(model, "tol", ϵ_tol)
# @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
# @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
# @constraint(model, c1, x_1 + x_2 >= 1.0)
# _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2)
# @objective(model, Min, _obj_CB(x_1, x_2))
# optimize!(model)
# println("Given (μ_0, μ_0^c, ω_1, ω_2) = ($μ_0, $μ_0_c, $ω_1, $ω_2)")
# println("Find the minimum of $(objective_value(model)) at (x_1,x_2) = ($(value(x_1)), $(value(x_2)))")

# #======================#
# # benchmark result - ν #
# #======================#
# ν_1_grid = collect(0.0:0.05:2.0)
# ν_1_size = length(ν_1_grid)
# ν_2_grid = collect(0.0:0.05:2.0)
# ν_2_size = length(ν_1_grid)
# ν_res = zeros(ν_1_size * ν_2_size, 8)
# ν_res_obj = zeros(ν_1_size, ν_2_size)
# ν_res_i = 1
# for ν_1_i = 1:ν_1_size, ν_2_i = 1:ν_2_size
#     # slove CB's optimization problem for a given μ_0 along with other benchmark parameters
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     set_attribute(model, "tol", ϵ_tol)
#     @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
#     @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
#     @constraint(model, c1, x_1 + x_2 >= 1.0)
#     _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1_grid[ν_1_i], ν_2_grid[ν_2_i])
#     @objective(model, Min, _obj_CB(x_1, x_2))
#     optimize!(model)
#     # save results
#     ν_res[ν_res_i, 1] = ν_1_grid[ν_1_i]
#     ν_res[ν_res_i, 2] = ν_2_grid[ν_2_i]
#     ν_res[ν_res_i, 3] = objective_value(model)
#     ν_res_obj[ν_1_i, ν_2_i] = objective_value(model)
#     ν_res[ν_res_i, 4] = value(x_1)
#     ν_res[ν_res_i, 5] = value(x_2)
#     ν_res[ν_res_i, 6] = μ_1(ν_res[ν_res_i, 4], ν_res[ν_res_i, 5], μ_0)
#     ν_res[ν_res_i, 7] = μ_2(ν_res[ν_res_i, 4], ν_res[ν_res_i, 5], μ_0)
#     ν_res[ν_res_i, 8] = 1.0 - δ * c(ν_res[ν_res_i, 4], ν_res[ν_res_i, 5], μ_0)
#     ν_res_i += 1
# end

# # rounding numbers
# ν_res = round.(ν_res, digits=4)

# # minimizer and minimum
# ν_min_i = argmin(ν_res[:, 3])
# ν_min = ν_res[ν_min_i, :]

# # heatmap
# fig = Figure(fontsize=32, size=(600, 500))
# ax = Axis(fig[1, 1], xlabel=L"$\nu_1$", ylabel=L"$\nu_2$")
# hm = heatmap!(ax, ν_1_grid, ν_2_grid, ν_res_obj, colormap=(:viridis,0.8))
# scatter!(ax, (ν_res[ν_min_i, 1], ν_res[ν_min_i, 2]), color=:red, strokecolor=:red, strokewidth=5)
# Colorbar(fig[:, end+1], hm)
# fig

# # save figures
# filename = "fig_optimal_ν" * ".pdf"
# save(PATH_FIG_γ * "\\" * filename, fig)
# filename = "fig_optimal_ν" * ".png"
# save(PATH_FIG_γ * "\\" * filename, fig)

#==============================#
# benchmark result - ν and μ_0 #
#==============================#
μ_0_grid = collect(0.1:0.1:0.90)
μ_0_size = length(μ_0_grid)
ν_1_grid = collect(0.0:0.1:2.0)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.0:0.1:2.0)
ν_2_size = length(ν_2_grid)
ν_res = zeros(μ_0_size, ν_1_size * ν_2_size, 8)
ν_res_obj = zeros(μ_0_size, ν_1_size, ν_2_size)
for μ_0_i = 1:μ_0_size
    ν_res_i = 1
    for ν_1_i = 1:ν_1_size, ν_2_i = 1:ν_2_size
        # slove CB's optimization problem for a given μ_0 along with other benchmark parameters
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", ϵ_tol)
        @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
        @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
        @constraint(model, c1, x_1 + x_2 >= 1.0)
        _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0_grid[μ_0_i], μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1_grid[ν_1_i], ν_2_grid[ν_2_i])
        @objective(model, Min, _obj_CB(x_1, x_2))
        optimize!(model)
        # save results
        ν_res[μ_0_i, ν_res_i, 1] = ν_1_grid[ν_1_i]
        ν_res[μ_0_i, ν_res_i, 2] = ν_2_grid[ν_2_i]
        ν_res[μ_0_i, ν_res_i, 3] = objective_value(model)
        ν_res_obj[μ_0_i, ν_1_i, ν_2_i] = objective_value(model)
        ν_res[μ_0_i, ν_res_i, 4] = value(x_1)
        ν_res[μ_0_i, ν_res_i, 5] = value(x_2)
        ν_res[μ_0_i, ν_res_i, 6] = μ_1(ν_res[μ_0_i, ν_res_i, 4], ν_res[μ_0_i, ν_res_i, 5], μ_0_grid[μ_0_i])
        ν_res[μ_0_i, ν_res_i, 7] = μ_2(ν_res[μ_0_i, ν_res_i, 4], ν_res[μ_0_i, ν_res_i, 5], μ_0_grid[μ_0_i])
        ν_res[μ_0_i, ν_res_i, 8] = 1.0 - δ * c(ν_res[μ_0_i, ν_res_i, 4], ν_res[μ_0_i, ν_res_i, 5], μ_0_grid[μ_0_i])
        ν_res_i += 1
    end
end

# rounding numbers
ν_res = round.(ν_res, digits=4)

# minimizer and minimum
ν_min_i = vec(getindex.(argmin(ν_res[:,:,3], dims=2), 2))
ν_1_min = [ν_res[μ_0_i, ν_min_i[μ_0_i], 1] for μ_0_i = 1:μ_0_size]
ν_2_min = [ν_res[μ_0_i, ν_min_i[μ_0_i], 2] for μ_0_i = 1:μ_0_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\mu_0$") # , ylabel=L"$\nu$"
lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_grid, ν_2_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=1, patchsize=(40,20))
fig

# save figures
filename = "fig_optimal_ν_μ_0" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, fig)
filename = "fig_optimal_ν_μ_0" * ".png"
save(PATH_FIG_γ * "\\" * filename, fig)

#================================#
# benchmark result - ν and μ_0_c #
#================================#
μ_0_c_grid = collect(0.1:0.1:0.90)
μ_0_c_size = length(μ_0_c_grid)
ν_1_grid = collect(0.0:0.1:2.0)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.0:0.1:2.0)
ν_2_size = length(ν_1_grid)
ν_res = zeros(μ_0_c_size, ν_1_size * ν_2_size, 8)
ν_res_obj = zeros(μ_0_c_size, ν_1_size, ν_2_size)
for μ_0_c_i = 1:μ_0_c_size
    ν_res_i = 1
    for ν_1_i = 1:ν_1_size, ν_2_i = 1:ν_2_size
        # slove CB's optimization problem for a given μ_0 along with other benchmark parameters
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", ϵ_tol)
        @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
        @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
        @constraint(model, c1, x_1 + x_2 >= 1.0)
        _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c_grid[μ_0_c_i], ω_1, ω_2, δ, γ, x_T, ν_1_grid[ν_1_i], ν_2_grid[ν_2_i])
        @objective(model, Min, _obj_CB(x_1, x_2))
        optimize!(model)
        # save results
        ν_res[μ_0_c_i, ν_res_i, 1] = ν_1_grid[ν_1_i]
        ν_res[μ_0_c_i, ν_res_i, 2] = ν_2_grid[ν_2_i]
        ν_res[μ_0_c_i, ν_res_i, 3] = objective_value(model)
        ν_res_obj[μ_0_c_i, ν_1_i, ν_2_i] = objective_value(model)
        ν_res[μ_0_c_i, ν_res_i, 4] = value(x_1)
        ν_res[μ_0_c_i, ν_res_i, 5] = value(x_2)
        ν_res[μ_0_c_i, ν_res_i, 6] = μ_1(ν_res[μ_0_c_i, ν_res_i, 4], ν_res[μ_0_c_i, ν_res_i, 5], μ_0_c_grid[μ_0_c_i])
        ν_res[μ_0_c_i, ν_res_i, 7] = μ_2(ν_res[μ_0_c_i, ν_res_i, 4], ν_res[μ_0_c_i, ν_res_i, 5], μ_0_c_grid[μ_0_c_i])
        ν_res[μ_0_c_i, ν_res_i, 8] = 1.0 - δ * c(ν_res[μ_0_c_i, ν_res_i, 4], ν_res[μ_0_c_i, ν_res_i, 5], μ_0_c_grid[μ_0_c_i])
        ν_res_i += 1
    end
end

# rounding numbers
ν_res = round.(ν_res, digits=4)

# minimizer and minimum
ν_min_i = vec(getindex.(argmin(ν_res[:,:,3], dims=2), 2))
ν_1_min = [ν_res[μ_0_c_i, ν_min_i[μ_0_c_i], 1] for μ_0_c_i = 1:μ_0_c_size]
ν_2_min = [ν_res[μ_0_c_i, ν_min_i[μ_0_c_i], 2] for μ_0_c_i = 1:μ_0_c_size]

# line plot
fig = Figure(fontsize=32, size=(600, 500))
ax = Axis(fig[1, 1], xlabel=L"$\mu^c_0$") # , ylabel=L"$\nu$"
lines!(ax, μ_0_c_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_c_grid, ν_1_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=:rt, nbanks=1, patchsize=(40,20))
fig

# save figures
filename = "fig_optimal_ν_μ_0_c" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, fig)
filename = "fig_optimal_ν_μ_0_c" * ".png"
save(PATH_FIG_γ * "\\" * filename, fig)