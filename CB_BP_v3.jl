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
δ = 1.0
ω_1 = 2.0
ω_2 = -2.0
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
μ_0_grid = collect(0.05:0.05:0.95)
μ_0_size = length(μ_0_grid)
ν_1_grid = collect(0.0:0.05:2.0)
ν_1_size = length(ν_1_grid)
ν_2_grid = collect(0.0:0.05:2.0)
ν_2_size = length(ν_1_grid)
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
ax = Axis(fig[1, 1], xlabel=L"$\mu_0$", ylabel=L"$\nu$")
lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_1$", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_grid, ν_1_min, label=L"$\nu_2$", color=:red, linestyle=:dash, linewidth=4)
fig

# save figures
filename = "fig_optimal_ν_μ_0" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, fig)
filename = "fig_optimal_ν_μ_0" * ".png"
save(PATH_FIG_γ * "\\" * filename, fig)

#=
#===========================#
# comparative statics - μ_0 #
#===========================#
μ_0_grid = collect(0.01:0.01:0.99)
μ_0_size = length(μ_0_grid)
μ_0_res = zeros(μ_0_size, 7)
μ_0_res_i = 1
for μ_0_i = 1:μ_0_size
    # slove CB's optimization problem for a given μ_0 along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0_grid[μ_0_i], μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    μ_0_res[μ_0_res_i, 1] = μ_0_grid[μ_0_i]
    μ_0_res[μ_0_res_i, 2] = objective_value(model)
    μ_0_res[μ_0_res_i, 3] = value(x_1)
    μ_0_res[μ_0_res_i, 4] = value(x_2)
    μ_0_res[μ_0_res_i, 5] = μ_1(μ_0_res[μ_0_res_i, 3], μ_0_res[μ_0_res_i, 4], μ_0_grid[μ_0_i])
    μ_0_res[μ_0_res_i, 6] = μ_2(μ_0_res[μ_0_res_i, 3], μ_0_res[μ_0_res_i, 4], μ_0_grid[μ_0_i])
    μ_0_res[μ_0_res_i, 7] = 1.0 - δ * c(μ_0_res[μ_0_res_i, 3], μ_0_res[μ_0_res_i, 4], μ_0_grid[μ_0_i])
    μ_0_res_i += 1
end

# rounding numbers
μ_0_res = round.(μ_0_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"HH prior $\mu_0$")
lines!(ax, μ_0_res[:, 1], μ_0_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_res[:, 1], μ_0_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "μ_0_" * "ω_1_$(ω_1)_" * "ω_2_$(ω_2)_" * "δ_$(δ)_" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
# f = Figure(fontsize=32, size=(600, 500))
# ax = Axis(f[1, 1], xlabel=L"HH prior $\mu_0$")
# lines!(ax, μ_0_res[:, 1], μ_0_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
# lines!(ax, μ_0_res[:, 1], μ_0_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
# lines!(ax, μ_0_res[:, 1], μ_0_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
# axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
# ylims!(-0.05, 1.05)
# filename = "fig_posterior_across_" * "μ_0" * ".pdf"
# save(PATH_FIG_γ * "\\" * filename, f)

# inflation surprise γ*(x_e(μ, x_T, ν_1, ν_2) - x_r(ω_i, x_T, ν_1, ν_2))
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"HH prior $\mu_0$")
lines!(ax, μ_0_res[:, 1], γ .* (x_e.(μ_0_res[:, 5], x_T, ν_1, ν_2) .- x_r(1, x_T, ν_1, ν_2)), label=L"(μ_1, \omega_1)", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_res[:, 1], γ .* (x_e.(μ_0_res[:, 5], x_T, ν_1, ν_2) .- x_r(2, x_T, ν_1, ν_2)), label=L"(μ_1, \omega_2)", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, μ_0_res[:, 1], γ .* (x_e.(μ_0_res[:, 6], x_T, ν_1, ν_2) .- x_r(1, x_T, ν_1, ν_2)), label=L"(μ_2, \omega_1)", color=:black, linestyle=:dot, linewidth=4)
lines!(ax, μ_0_res[:, 1], γ .* (x_e.(μ_0_res[:, 6], x_T, ν_1, ν_2) .- x_r(2, x_T, ν_1, ν_2)), label=L"(μ_2, \omega_2)", color=:green, linestyle=:dashdot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
filename = "fig_posterior_across_" * "μ_0_" * "ω_1_$(ω_1)_" * "ω_2_$(ω_2)_" * "δ_$(δ)_" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#=============================#
# comparative statics - μ_0_c #
#=============================#
# δ = 1.0
# μ_0 = 0.1
# ω_1 = 2.0
μ_0_c_grid = collect(0.01:0.01:0.99)
μ_0_c_size = length(μ_0_c_grid)
μ_0_c_res = zeros(μ_0_c_size, 7)
μ_0_c_res_i = 1
for μ_0_c_i = 1:μ_0_c_size
    # slove CB's optimization problem for a given μ_0_c along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c_grid[μ_0_c_i], ω_1, ω_2, δ, γ, x_T, ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    μ_0_c_res[μ_0_c_res_i, 1] = μ_0_c_grid[μ_0_c_i]
    μ_0_c_res[μ_0_c_res_i, 2] = objective_value(model)
    μ_0_c_res[μ_0_c_res_i, 3] = value(x_1)
    μ_0_c_res[μ_0_c_res_i, 4] = value(x_2)
    μ_0_c_res[μ_0_c_res_i, 5] = μ_1(μ_0_c_res[μ_0_c_res_i, 3], μ_0_c_res[μ_0_c_res_i, 4], μ_0)
    μ_0_c_res[μ_0_c_res_i, 6] = μ_2(μ_0_c_res[μ_0_c_res_i, 3], μ_0_c_res[μ_0_c_res_i, 4], μ_0)
    μ_0_c_res[μ_0_c_res_i, 7] = 1.0 - δ * c(μ_0_c_res[μ_0_c_res_i, 3], μ_0_c_res[μ_0_c_res_i, 4], μ_0)
    μ_0_c_res_i += 1
end

# rounding numbers
μ_0_c_res = round.(μ_0_c_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"CB prior $\mu_0^c$")
lines!(ax, μ_0_c_res[:, 1], μ_0_c_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_c_res[:, 1], μ_0_c_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "μ_0_c_" * "ω_1_$(ω_1)_" * "ω_2_$(ω_2)_" * "δ_$(δ)_" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
# f = Figure(fontsize=32, size=(600, 500))
# ax = Axis(f[1, 1], xlabel=L"CB prior $\mu_0^c$")
# lines!(ax, μ_0_c_res[:, 1], μ_0_c_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
# lines!(ax, μ_0_c_res[:, 1], μ_0_c_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
# lines!(ax, μ_0_c_res[:, 1], μ_0_c_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
# axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))

# ylims!(-0.05, 1.05)
# filename = "fig_posterior_across_" * "μ_0_c" * ".pdf"
# save(PATH_FIG_γ * "\\" * filename, f)

# inflation surprise γ*(x_e(μ, x_T, ν_1, ν_2) - x_r(ω_i, x_T, ν_1, ν_2))
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"CB prior $\mu_0^c$")
lines!(ax, μ_0_c_res[:, 1], γ .* (x_e.(μ_0_c_res[:, 5], x_T, ν_1, ν_2) .- x_r(1, x_T, ν_1, ν_2)), label=L"(μ_1, \omega_1)", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, μ_0_c_res[:, 1], γ .* (x_e.(μ_0_c_res[:, 5], x_T, ν_1, ν_2) .- x_r(2, x_T, ν_1, ν_2)), label=L"(μ_1, \omega_2)", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, μ_0_c_res[:, 1], γ .* (x_e.(μ_0_c_res[:, 6], x_T, ν_1, ν_2) .- x_r(1, x_T, ν_1, ν_2)), label=L"(μ_2, \omega_1)", color=:black, linestyle=:dot, linewidth=4)
lines!(ax, μ_0_c_res[:, 1], γ .* (x_e.(μ_0_c_res[:, 6], x_T, ν_1, ν_2) .- x_r(2, x_T, ν_1, ν_2)), label=L"(μ_2, \omega_2)", color=:green, linestyle=:dashdot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
filename = "fig_posterior_across_" * "μ_0_c_" * "ω_1_$(ω_1)_" * "ω_2_$(ω_2)_" * "δ_$(δ)_" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#=========================#
# comparative statics - δ #
#=========================#
δ_grid = collect(0.00:0.01:1.00)
δ_size = length(δ_grid)
δ_res = zeros(δ_size, 7)
δ_res_i = 1
for δ_i = 1:δ_size
    # slove CB's optimization problem for a given δ along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ_grid[δ_i], γ, x_T, ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    δ_res[δ_res_i, 1] = δ_grid[δ_i]
    δ_res[δ_res_i, 2] = objective_value(model)
    δ_res[δ_res_i, 3] = value(x_1)
    δ_res[δ_res_i, 4] = value(x_2)
    δ_res[δ_res_i, 5] = μ_1(δ_res[δ_res_i, 3], δ_res[δ_res_i, 4], μ_0)
    δ_res[δ_res_i, 6] = μ_2(δ_res[δ_res_i, 3], δ_res[δ_res_i, 4], μ_0)
    δ_res[δ_res_i, 7] = 1.0 - δ_res[δ_res_i, 1] * c(δ_res[δ_res_i, 3], δ_res[δ_res_i, 4], μ_0)
    δ_res_i += 1
end

# rounding numbers
δ_res = round.(δ_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Share of naive HH $\delta$")
lines!(ax, δ_res[:, 1], δ_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, δ_res[:, 1], δ_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "δ" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Share of naive HH $\delta$")
lines!(ax, δ_res[:, 1], δ_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, δ_res[:, 1], δ_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, δ_res[:, 1], δ_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_posterior_across_" * "δ" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#===========================#
# comparative statics - ω_1 #
#===========================#
ω_1_grid = collect(0.50:0.01:1.50)
ω_1_size = length(ω_1_grid)
ω_1_res = zeros(ω_1_size, 7)
ω_1_res_i = 1
for ω_1_i = 1:ω_1_size
    # slove CB's optimization problem for a given ω_1 along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1_grid[ω_1_i], ω_2, δ, γ, x_T, ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    ω_1_res[ω_1_res_i, 1] = ω_1_grid[ω_1_i]
    ω_1_res[ω_1_res_i, 2] = objective_value(model)
    ω_1_res[ω_1_res_i, 3] = value(x_1)
    ω_1_res[ω_1_res_i, 4] = value(x_2)
    ω_1_res[ω_1_res_i, 5] = μ_1(ω_1_res[ω_1_res_i, 3], ω_1_res[ω_1_res_i, 4], μ_0)
    ω_1_res[ω_1_res_i, 6] = μ_2(ω_1_res[ω_1_res_i, 3], ω_1_res[ω_1_res_i, 4], μ_0)
    ω_1_res[ω_1_res_i, 7] = 1.0 - δ * c(ω_1_res[ω_1_res_i, 3], ω_1_res[ω_1_res_i, 4], μ_0)
    ω_1_res_i += 1
end

# rounding numbers
ω_1_res = round.(ω_1_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Bad shock $\omega_1$")
lines!(ax, ω_1_res[:, 1], ω_1_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_1_res[:, 1], ω_1_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "ω_1" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Bad shock $\omega_1$")
lines!(ax, ω_1_res[:, 1], ω_1_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_1_res[:, 1], ω_1_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, ω_1_res[:, 1], ω_1_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_posterior_across_" * "ω_1" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#===========================#
# comparative statics - ω_2 #
#===========================#
ω_2_grid = collect(-0.50:-0.01:-1.50)
ω_2_size = length(ω_2_grid)
ω_2_res = zeros(ω_2_size, 7)
ω_2_res_i = 1
for ω_2_i = 1:ω_2_size
    # slove CB's optimization problem for a given ω_2 along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2_grid[ω_2_i], δ, γ, x_T, ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    ω_2_res[ω_2_res_i, 1] = ω_2_grid[ω_2_i]
    ω_2_res[ω_2_res_i, 2] = objective_value(model)
    ω_2_res[ω_2_res_i, 3] = value(x_1)
    ω_2_res[ω_2_res_i, 4] = value(x_2)
    ω_2_res[ω_2_res_i, 5] = μ_1(ω_2_res[ω_2_res_i, 3], ω_2_res[ω_2_res_i, 4], μ_0)
    ω_2_res[ω_2_res_i, 6] = μ_2(ω_2_res[ω_2_res_i, 3], ω_2_res[ω_2_res_i, 4], μ_0)
    ω_2_res[ω_2_res_i, 7] = 1.0 - δ * c(ω_2_res[ω_2_res_i, 3], ω_2_res[ω_2_res_i, 4], μ_0)
    ω_2_res_i += 1
end

# rounding numbers
ω_2_res = round.(ω_2_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Good shock $\omega_2$")
lines!(ax, ω_2_res[:, 1], ω_2_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_2_res[:, 1], ω_2_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "ω_2" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Good shock $\omega_2$")
lines!(ax, ω_2_res[:, 1], ω_2_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ω_2_res[:, 1], ω_2_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, ω_2_res[:, 1], ω_2_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_posterior_across_" * "ω_2" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#=========================#
# comparative statics - γ #
#=========================#
γ_grid = collect(0.1:0.01:5)
γ_size = length(γ_grid)
γ_res = zeros(γ_size, 7)
γ_res_i = 1
for γ_i = 1:γ_size
    # slove CB's optimization problem for a given γ along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 1.0 - ϵ_x)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 1.0 - ϵ_x)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ_grid[γ_i], x_T, ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    γ_res[γ_res_i, 1] = γ_grid[γ_i]
    γ_res[γ_res_i, 2] = objective_value(model)
    γ_res[γ_res_i, 3] = value(x_1)
    γ_res[γ_res_i, 4] = value(x_2)
    γ_res[γ_res_i, 5] = μ_1(γ_res[γ_res_i, 3], γ_res[γ_res_i, 4], μ_0)
    γ_res[γ_res_i, 6] = μ_2(γ_res[γ_res_i, 3], γ_res[γ_res_i, 4], μ_0)
    γ_res[γ_res_i, 7] = 1.0 - δ * c(γ_res[γ_res_i, 3], γ_res[γ_res_i, 4], μ_0)
    γ_res_i += 1
end

# rounding numbers
γ_res = round.(γ_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Inflation suprise $\gamma$")
lines!(ax, γ_res[:, 1], γ_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, γ_res[:, 1], γ_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "γ" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Inflation suprise $\gamma$")
lines!(ax, γ_res[:, 1], γ_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, γ_res[:, 1], γ_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, γ_res[:, 1], γ_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_posterior_across_" * "γ" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#=========================#
# comparative statics - ν #
#=========================#
ν_grid = collect(0.50:0.01:1.50)
ν_size = length(ν_grid)
ν_res = zeros(ν_size, 7)
ν_res_i = 1
for ν_i = 1:ν_size
    # slove CB's optimization problem for a given ν along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T, ν_grid[ν_i], ν_grid[ν_i])
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    ν_res[ν_res_i, 1] = ν_grid[ν_i]
    ν_res[ν_res_i, 2] = objective_value(model)
    ν_res[ν_res_i, 3] = value(x_1)
    ν_res[ν_res_i, 4] = value(x_2)
    ν_res[ν_res_i, 5] = μ_1(ν_res[ν_res_i, 3], ν_res[ν_res_i, 4], μ_0)
    ν_res[ν_res_i, 6] = μ_2(ν_res[ν_res_i, 3], ν_res[ν_res_i, 4], μ_0)
    ν_res[ν_res_i, 7] = 1.0 - δ * c(ν_res[ν_res_i, 3], ν_res[ν_res_i, 4], μ_0)
    ν_res_i += 1
end

# rounding numbers
ν_res = round.(ν_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Inflation flexibility $\nu$")
lines!(ax, ν_res[:, 1], ν_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ν_res[:, 1], ν_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "ν" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Inflation flexibility $\nu$")
lines!(ax, ν_res[:, 1], ν_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, ν_res[:, 1], ν_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, ν_res[:, 1], ν_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_posterior_across_" * "ν" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

#===========================#
# comparative statics - x_T #
#===========================#
x_T_grid = collect(1.50:0.01:2.50)
x_T_size = length(x_T_grid)
x_T_res = zeros(x_T_size, 7)
x_T_res_i = 1
for x_T_i = 1:x_T_size
    # slove CB's optimization problem for a given x_T along with other benchmark parameters
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "tol", ϵ_tol)
    @variable(model, ϵ_x <= x_1 <= (1.0 - ϵ_x), start = 0.5)
    @variable(model, ϵ_x <= x_2 <= (1.0 - ϵ_x), start = 0.5)
    @constraint(model, c1, x_1 + x_2 >= 1.0)
    _obj_CB(x_1, x_2) = obj_CB(x_1, x_2, μ_0, μ_0_c, ω_1, ω_2, δ, γ, x_T_grid[x_T_i], ν_1, ν_2)
    @objective(model, Min, _obj_CB(x_1, x_2))
    optimize!(model)
    # save results
    x_T_res[x_T_res_i, 1] = x_T_grid[x_T_i]
    x_T_res[x_T_res_i, 2] = objective_value(model)
    x_T_res[x_T_res_i, 3] = value(x_1)
    x_T_res[x_T_res_i, 4] = value(x_2)
    x_T_res[x_T_res_i, 5] = μ_1(x_T_res[x_T_res_i, 3], x_T_res[x_T_res_i, 4], μ_0)
    x_T_res[x_T_res_i, 6] = μ_2(x_T_res[x_T_res_i, 3], x_T_res[x_T_res_i, 4], μ_0)
    x_T_res[x_T_res_i, 7] = 1.0 - δ * c(x_T_res[x_T_res_i, 3], x_T_res[x_T_res_i, 4], μ_0)
    x_T_res_i += 1
end

# rounding numbers
x_T_res = round.(x_T_res, digits=4)

# optimal information disclosure π = (x_1, x_2)
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Inflation target $x_T$")
lines!(ax, x_T_res[:, 1], x_T_res[:, 3], label=L"x_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, x_T_res[:, 1], x_T_res[:, 4], label=L"x_2", color=:red, linestyle=:dash, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_optimal_π_across_" * "x_T" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)

# posterior μ_s_1 and μ_s_1 
f = Figure(fontsize=32, size=(600, 500))
ax = Axis(f[1, 1], xlabel=L"Inflation target $x_T$")
lines!(ax, x_T_res[:, 1], x_T_res[:, 5], label=L"\mu_1", color=:blue, linestyle=nothing, linewidth=4)
lines!(ax, x_T_res[:, 1], x_T_res[:, 6], label=L"\mu_2", color=:red, linestyle=:dash, linewidth=4)
lines!(ax, x_T_res[:, 1], x_T_res[:, 7], label=L"1-\delta c(\pi)", color=:black, linestyle=:dot, linewidth=4)
axislegend(position=(0.95, 0.05), nbanks=2, patchsize=(40,20))
ylims!(-0.05, 1.05)
filename = "fig_posterior_across_" * "x_T" * ".pdf"
save(PATH_FIG_γ * "\\" * filename, f)
=#