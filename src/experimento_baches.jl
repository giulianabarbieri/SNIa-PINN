#!/usr/bin/env julia

"""
================================================================================
Experimento de Baches: PINN (Arnett + causal) vs Gaussian Process
================================================================================

Simula huecos observacionales realistas sobre ZTF25aavdmzf (168 pts) y compara
la reconstrucción de la PINN contra un GP estándar con kernel RBF.

Métricas: NRMSE y R² en los puntos excluidos de cada truncamiento.
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Random
using Printf
using Plots
using Dates

include("training_inverse.jl")

# ==============================================================================
# 0. CONFIGURACIÓN
# ==============================================================================

const ARCHIVO_COMPLETO = joinpath(@__DIR__, "..", "dataset", "ZTF25aavdmzf_detections.csv")
const Z_OBJETO = 0.0407
const PARAMS_OPTIMOS = (t0_init = -0.05f0, s1_epochs = 2000, s2_epochs = 3000, ε_causal = 0.3f0)
const SEED = 123

Random.seed!(SEED)

# ── Carga de datos completos ──────────────────────────────────────────────────
df_raw = CSV.read(ARCHIVO_COMPLETO, DataFrame)
df_clean = filter(r -> !r.dubious .&& !isnan(r.magpsf) .&& r.magpsf > 0, df_raw)
sort!(df_clean, :mjd)

H0       = 70.0
c_light  = 299792.458
D_Mpc    = (c_light * Z_OBJETO) / H0
t_min_gt = minimum(df_clean.mjd)
T_MAX_GT = Float32(maximum(df_clean.mjd) - t_min_gt)

function mag_to_lum(mag)
    if mag <= 0.0 || isnan(mag); return 0.0; end
    mag_abs = mag - 5.0 * log10(D_Mpc) - 25.0
    return 3.0128e35 * (10.0^min(-0.4 * mag_abs, 50.0))
end

t_full_dias = Float32.(df_clean.mjd .- t_min_gt)
t_full_norm = Float32.(t_full_dias ./ T_MAX_GT)
lums_full   = mag_to_lum.(df_clean.magpsf)
L_MAX_GT    = maximum(filter(x -> !isinf(x) && !isnan(x), lums_full))
flux_full   = Float32.([isnan(x/L_MAX_GT) ? 0.0f0 : Float32(x/L_MAX_GT) for x in lums_full])
flux_range  = maximum(flux_full) - minimum(flux_full)
fids_full   = Float32.(df_clean.fid)

println("📂 Datos completos: $(length(t_full_norm)) pts | T_MAX=$(T_MAX_GT)d | L_MAX=$(L_MAX_GT) erg/s\n")

# ==============================================================================
# 1. GP DESDE CERO (RBF + grid search sobre log-marginal-likelihood)
# ==============================================================================

function gp_fit_predict(t_train, y_train, t_pred)
    n = length(t_train)
    l_grid   = Float32[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    s2_grid  = Float32[0.1, 0.5, 1.0, 2.0, 5.0]
    sn2_grid = Float32[1e-6, 1e-4, 1e-3, 1e-2, 0.1]

    function rbf(X1, X2, l, s2)
        K = zeros(Float32, length(X1), length(X2))
        inv2l2 = Float32(0.5) / (l * l)
        for i in 1:length(X1), j in 1:length(X2)
            d = X1[i] - X2[j]
            K[i,j] = s2 * exp(-d * d * inv2l2)
        end
        return K
    end

    best_nll, best_l, best_s2, best_sn2 = Inf, Float32(0.1), Float32(1.0), Float32(1e-3)
    for l in l_grid, s2 in s2_grid, sn2 in sn2_grid
        K = rbf(t_train, t_train, l, s2) + sn2 * I
        try
            L = cholesky(Symmetric(K))
            alpha = L \ y_train
            nll = Float32(0.5) * (y_train' * alpha) + sum(log.(diag(L.L))) + Float32(0.5) * n * log(Float32(2pi))
            if nll < best_nll; best_nll=nll; best_l=l; best_s2=s2; best_sn2=sn2; end
        catch; end
    end

    K_tr = rbf(t_train, t_train, best_l, best_s2) + best_sn2 * I
    L = cholesky(Symmetric(K_tr))
    alpha = L \ y_train
    K_star = rbf(t_pred, t_train, best_l, best_s2)
    mu = K_star * alpha
    return mu
end

# ==============================================================================
# 2. TRUNCAMIENTO
# ==============================================================================

function guardar_csv_truncado(df, path)
    mkpath(dirname(path))
    CSV.write(path, DataFrame(
        mjd=df.mjd, fid=df.fid, magpsf=df.magpsf,
        sigmapsf=df.sigmapsf, dubious=df.dubious))
    return path
end

function truncar(df_full, tipo, args; seed)
    rng = MersenneTwister(seed)
    mjd = df_full.mjd
    mjd_rng = maximum(mjd) - minimum(mjd)

    if tipo == :random
        frac = args[1]
        keep = rand(rng, nrow(df_full)) .< frac
        return df_full[keep, :]
    elseif tipo == :gaps
        gaps = args[1]  # vector de duraciones en días
        mask = trues(nrow(df_full))
        centers = minimum(mjd) .+ rand(rng, length(gaps)) .* mjd_rng
        for (c, w) in zip(centers, gaps)
            h = w/2
            mask .&= .!((mjd .>= c-h) .& (mjd .<= c+h))
        end
        return df_full[mask, :]
    elseif tipo == :combinado
        frac = args[1]; gaps = args[2]
        mask = trues(nrow(df_full))
        centers = minimum(mjd) .+ rand(rng, length(gaps)) .* mjd_rng
        for (c, w) in zip(centers, gaps)
            h = w/2
            mask .&= .!((mjd .>= c-h) .& (mjd .<= c+h))
        end
        df_gaps = df_full[mask, :]
        keep = rand(rng, nrow(df_gaps)) .< frac
        return df_gaps[keep, :]
    end
    return df_full
end

# ==============================================================================
# 3. EJECUCIÓN DE EXPERIMENTOS
# ==============================================================================

experimentos = [
    (nombre="Random 50%",  tipo=:random,    args=(0.50f0,)),
    (nombre="Random 80%",  tipo=:random,    args=(0.20f0,)),
    (nombre="Gaps 3×8d",   tipo=:gaps,      args=([8.0, 8.0, 8.0],)),
    (nombre="Gaps 2×15d",  tipo=:gaps,      args=([15.0, 15.0],)),
    (nombre="Combinado",   tipo=:combinado,  args=(0.70f0, [10.0, 10.0, 8.0])),
    (nombre="Extremo",     tipo=:combinado,  args=(0.30f0, [12.0, 14.0, 10.0])),
]

# ── Ground truth: PINN con datos completos ────────────────────────────────────
println("🏆 Entrenando PINN ground truth con datos completos...")
flush(stdout)
r_gt = run_pinn_experiment(
    archivo=ARCHIVO_COMPLETO, z=Z_OBJETO, t0_init=PARAMS_OPTIMOS.t0_init,
    lit_ni_range="0.52 – 0.58", s1_epochs=PARAMS_OPTIMOS.s1_epochs,
    s2_epochs=PARAMS_OPTIMOS.s2_epochs, ε_causal=PARAMS_OPTIMOS.ε_causal,
    output_prefix="gt_completo")
M_Ni0_gt = r_gt.M_Ni0_final
println("   GT: M_Ni0=$(round(M_Ni0_gt,digits=4)), t₀=$(round(r_gt.t0_final_dias,digits=2))d\n")
flush(stdout)

# Predicción PINN GT en grilla densa
t_grid = Float32.(range(0.0f0, 1.0f0, length=300))
pinn_gt_g = Float32[]
pinn_gt_r = Float32[]
for tp in t_grid
    raw_g, _ = r_gt.model_flux([tp; 1.0f0], r_gt.ps_total.beta1, r_gt.st_flux)
    raw_r, _ = r_gt.model_flux([tp; 2.0f0], r_gt.ps_total.beta1, r_gt.st_flux)
    push!(pinn_gt_g, apply_hard_constraint(tp, raw_g[1], r_gt.t0_final))
    push!(pinn_gt_r, apply_hard_constraint(tp, raw_r[1], r_gt.t0_final))
end
pinn_gt_avg = (pinn_gt_g .+ pinn_gt_r) ./ 2.0f0

resultados = DataFrame(
    experimento=String[], tipo=String[], n_trunc=Int[], n_excl=Int[],
    NRMSE_GP=Float32[], NRMSE_PINN=Float32[],
    R2_GP=Float32[], R2_PINN=Float32[],
    ΔM_Ni0=Float32[], M_Ni0_GT=Float32[], M_Ni0_PINN=Float32[],
)
plots_list = []

for (idx, exp_cfg) in enumerate(experimentos)
    seed_e = SEED + idx*100
    println("$(repeat("=",70))")
    println("📐 Exp $idx/$(length(experimentos)): $(exp_cfg.nombre)")
    println("$(repeat("=",70))")
    flush(stdout)

    # ── Truncar ──────────────────────────────────────────────────────────
    df_trunc = truncar(df_clean, exp_cfg.tipo, exp_cfg.args; seed=seed_e)
    path_trunc = guardar_csv_truncado(df_trunc, "/tmp/exp_baches/trunc_$(idx).csv")

    # Identificar excluidos
    mjd_trunc = Set(df_trunc.mjd)
    excl_idx  = findall(m -> !(m in mjd_trunc), df_clean.mjd)
    t_excl    = t_full_norm[excl_idx]
    f_excl    = flux_full[excl_idx]
    n_excl    = length(excl_idx)
    println("   Pts: $(nrow(df_trunc)) retenidos, $n_excl excluidos")

    # ── PINN sobre truncado ──────────────────────────────────────────────
    r_pinn = run_pinn_experiment(
        archivo=path_trunc, z=Z_OBJETO, t0_init=PARAMS_OPTIMOS.t0_init,
        lit_ni_range="0.52 – 0.58", s1_epochs=PARAMS_OPTIMOS.s1_epochs,
        s2_epochs=PARAMS_OPTIMOS.s2_epochs, ε_causal=PARAMS_OPTIMOS.ε_causal,
        output_prefix="bache_$(exp_cfg.nombre)")

    # Predicción PINN en grilla (bandas g y r separadas)
    pinn_g = Float32[]
    pinn_r = Float32[]
    for tp in t_grid
        raw_g, _ = r_pinn.model_flux([tp; 1.0f0], r_pinn.ps_total.beta1, r_pinn.st_flux)
        raw_r, _ = r_pinn.model_flux([tp; 2.0f0], r_pinn.ps_total.beta1, r_pinn.st_flux)
        push!(pinn_g, apply_hard_constraint(tp, raw_g[1], r_pinn.t0_final))
        push!(pinn_r, apply_hard_constraint(tp, raw_r[1], r_pinn.t0_final))
    end

    function interpolar(t_val, t_grid, y_grid)
        return y_grid[argmin(abs.(t_grid .- t_val))]
    end

    # ── Datos truncados por banda ────────────────────────────────────────
    t_trunc_norm = Float32.((df_trunc.mjd .- t_min_gt) ./ T_MAX_GT)
    flux_trunc_norm = Float32.([isnan(x/L_MAX_GT) ? 0.0f0 : Float32(x/L_MAX_GT)
                                 for x in mag_to_lum.(df_trunc.magpsf)])

    # ── BUCLE POR BANDA (g=1, r=2) ──────────────────────────────────────
    bandas = [(1.0f0, "g", "Banda g", :green,  :darkgreen),
              (2.0f0, "r", "Banda r", :red,    :darkred)]

    nrmse_g, r2_g, nrmse_r, r2_r = NaN32, NaN32, NaN32, NaN32
    gp_g, gp_r, pinn_excl_g, pinn_excl_r = nothing, nothing, nothing, nothing
    f_excl_g, f_excl_r, t_excl_g, t_excl_r = nothing, nothing, nothing, nothing

    for (fid, btag, blabel, bcolor, bcolor_dark) in bandas
        # ── Datos de esta banda ───────────────────────────────────────
        mask_trunc_band = df_trunc.fid .== fid
        t_trunc_b = t_trunc_norm[mask_trunc_band]
        f_trunc_b = flux_trunc_norm[mask_trunc_band]

        # Puntos excluidos de esta banda
        mask_full_band = fids_full .== fid
        mjd_full_band = df_clean.mjd[mask_full_band]
        excl_b_idx = findall(m -> !(m in Set(df_trunc.mjd)), mjd_full_band)
        t_excl_b = t_full_norm[mask_full_band][excl_b_idx]
        f_excl_b = flux_full[mask_full_band][excl_b_idx]

        # ── PINN en puntos excluidos ──────────────────────────────────
        pinn_excl_b = Float32[interpolar(t, t_grid, fid == 1.0f0 ? pinn_g : pinn_r) for t in t_excl_b]

        # ── GP ────────────────────────────────────────────────────────
        if length(t_trunc_b) < 3
            continue  # pocos puntos para GP
        end
        gp_pred_band = gp_fit_predict(t_trunc_b, f_trunc_b, t_grid)
        gp_excl_b = Float32[interpolar(t, t_grid, gp_pred_band) for t in t_excl_b]

        # ── Métricas por banda ────────────────────────────────────────
        rmse_gp_b   = sqrt(mean((gp_excl_b .- f_excl_b).^2))
        rmse_pinn_b = sqrt(mean((pinn_excl_b .- f_excl_b).^2))
        nrmse_gp_b  = rmse_gp_b / max(flux_range, 1.0f-8) * 100
        nrmse_pinn_b = rmse_pinn_b / max(flux_range, 1.0f-8) * 100
        ss_tot_b = sum((f_excl_b .- mean(f_excl_b)).^2)
        r2_gp_b    = 1.0f0 - sum((gp_excl_b .- f_excl_b).^2)   / max(ss_tot_b, 1.0f-8)
        r2_pinn_b  = 1.0f0 - sum((pinn_excl_b .- f_excl_b).^2) / max(ss_tot_b, 1.0f-8)

        @printf("   Banda %s: NRMSE GP=%.2f%% PINN=%.2f%% | R² GP=%.3f PINN=%.3f\n",
                btag, nrmse_gp_b, nrmse_pinn_b, r2_gp_b, r2_pinn_b)

        if fid == 1.0f0
            nrmse_g, r2_g, gp_g, pinn_excl_g = nrmse_pinn_b, r2_pinn_b, gp_pred_band, pinn_excl_b
            f_excl_g, t_excl_g = f_excl_b, t_excl_b
        else
            nrmse_r, r2_r, gp_r, pinn_excl_r = nrmse_pinn_b, r2_pinn_b, gp_pred_band, pinn_excl_b
            f_excl_r, t_excl_r = f_excl_b, t_excl_b
        end
    end

    # ── Métricas combinadas ──────────────────────────────────────────────
    nrmse_pinn_avg = (nrmse_g + nrmse_r) / 2.0f0
    r2_pinn_avg    = (r2_g + r2_r) / 2.0f0
    ΔM_Ni0 = abs(r_pinn.M_Ni0_final - M_Ni0_gt) / max(M_Ni0_gt, 1.0f-8) * 100

    @printf("   ▸ NRMSE PINN avg: %.2f%% | R² PINN avg: %.3f | ΔM_Ni0: %.2f%%\n",
            nrmse_pinn_avg, r2_pinn_avg, ΔM_Ni0)
    flush(stdout)

    push!(resultados, (exp_cfg.nombre, string(exp_cfg.tipo), nrow(df_trunc),
                       n_excl, NaN32, nrmse_pinn_avg, NaN32, r2_pinn_avg,
                       ΔM_Ni0, M_Ni0_gt, r_pinn.M_Ni0_final))

    # ── Plot por banda ───────────────────────────────────────────────────
    # Panel g
    mask_trunc_g = df_trunc.fid .== 1.0f0
    p_g = scatter(t_trunc_norm[mask_trunc_g], flux_trunc_norm[mask_trunc_g],
        label="Train g ($(sum(mask_trunc_g)) pts)", color=:green,
        markersize=3, markerstrokewidth=0, legend=:topright,
        title="$(exp_cfg.nombre) — Banda g\nPINN NRMSE=$(round(nrmse_g,digits=1))% R²=$(round(r2_g,digits=3))",
        xlabel="t_norm", ylabel="Flux", dpi=120, legendfontsize=7, titlefontsize=9)
    if !isnothing(t_excl_g) && length(t_excl_g) > 0
        scatter!(p_g, t_excl_g, f_excl_g, label="Excluidos GT",
                 color=:gray, markersize=4, marker=:xcross, alpha=0.6)
    end
    plot!(p_g, t_grid, pinn_g, label="PINN", color=:darkgreen, linewidth=2)
    if !isnothing(gp_g)
        plot!(p_g, t_grid, gp_g, label="GP", color=:blue, linewidth=2, linestyle=:dash)
    end

    # Panel r
    mask_trunc_r = df_trunc.fid .== 2.0f0
    p_r = scatter(t_trunc_norm[mask_trunc_r], flux_trunc_norm[mask_trunc_r],
        label="Train r ($(sum(mask_trunc_r)) pts)", color=:red,
        markersize=3, markerstrokewidth=0, legend=:topright,
        title="$(exp_cfg.nombre) — Banda r\nPINN NRMSE=$(round(nrmse_r,digits=1))% R²=$(round(r2_r,digits=3))",
        xlabel="t_norm", ylabel="Flux", dpi=120, legendfontsize=7, titlefontsize=9)
    if !isnothing(t_excl_r) && length(t_excl_r) > 0
        scatter!(p_r, t_excl_r, f_excl_r, label="Excluidos GT",
                 color=:gray, markersize=4, marker=:xcross, alpha=0.6)
    end
    plot!(p_r, t_grid, pinn_r, label="PINN", color=:darkred, linewidth=2)
    if !isnothing(gp_r)
        plot!(p_r, t_grid, gp_r, label="GP", color=:blue, linewidth=2, linestyle=:dash)
    end

    push!(plots_list, p_g)
    push!(plots_list, p_r)
end

# ==============================================================================
# 4. GUARDAR Y MOSTRAR RESULTADOS
# ==============================================================================

CSV.write("resultados_baches.csv", resultados)

println("\n" * "═"^95)
println("  RESUMEN DEL EXPERIMENTO DE BACHES — PINN vs GP")
println("═"^95)
@printf("  %-18s | %6s | %5s | %8s %8s | %6s %6s | %8s\n",
        "Experimento", "n_trunc", "n_excl", "NRMSE_GP", "NRMSE_PINN", "R2_GP", "R2_PINN", "ΔM_Ni0")
println("-"^95)
for row in eachrow(resultados)
    @printf("  %-18s | %6d | %5d | %7.2f%% %7.2f%% | %5.3f %5.3f | %7.2f%%\n",
            row.experimento, row.n_trunc, row.n_excl,
            row.NRMSE_GP, row.NRMSE_PINN, row.R2_GP, row.R2_PINN,
            row.ΔM_Ni0)
end
println("═"^95)

# Gráfico consolidado (12 paneles: 6 exp × 2 bandas)
nrows = ceil(Int, length(plots_list) / 3)
plt = plot(plots_list..., layout=(nrows, 3), size=(1500, 380*nrows))
savefig(plt, "experimento_baches.png")
println("\n📊 Gráfico guardado en: experimento_baches.png")
println("📄 Resultados guardados en: resultados_baches.csv")
println("✅ Experimento completado.")
