#!/usr/bin/env julia

"""
================================================================================
Grid Search: Exploración de estabilidad de t₀ y convergencia de M_Ni0
================================================================================

Barre 3 objetos ZTF × 4 valores iniciales de t0_explosion × 2 regímenes de épocas
(24 corridas totales). Guarda todas las métricas en un CSV consolidado.
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Printf
using Dates

include("training_inverse.jl")

# ==============================================================================
# CONFIGURACIÓN DEL GRID
# ==============================================================================

OBJETOS = [
    (nombre    = "ZTF25aavdmzf",
     archivo   = joinpath(@__DIR__, "..", "dataset", "ZTF25aavdmzf_detections.csv"),
     z         = 0.0407,
     lit_ni    = "0.52 – 0.58"),

    (nombre    = "ZTF25aaxjntk",
     archivo   = joinpath(@__DIR__, "..", "dataset", "ZTF25aaxjntk_detections.csv"),
     z         = 0.01633,
     lit_ni    = "0.35 – 0.42"),

    (nombre    = "ZTF25aaxeojh",
     archivo   = joinpath(@__DIR__, "..", "dataset", "ZTF25aaxeojh_detections.csv"),
     z         = 0.049, #correccion del redshift!!!
    lit_ni    = "~0.80 – 1.30"),
]

T0_VALUES     = Float32[-0.05]
S1_EPOCHS     = [2000]
S2_EPOCHS     = 3000
EPS_CAUSAL_VALUES = Float32[0.3]

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

results = DataFrame(
    objeto           = String[],
    z                = Float64[],
    t0_inicial       = Float32[],
    s1_epochs        = Int[],
    s2_epochs        = Int[],
    ε_causal         = Float32[],
    t0_final_norm    = Float32[],
    t0_final_dias    = Float32[],
    M_Ni0_PINN       = Float32[],
    lit_ni           = String[],
    κ_promedio       = Float32[],
    rise_time_pinn   = Float32[],
    loss_final       = Float32[],
    λ_final          = Float32[],
    T_MAX            = Float32[],
    n_obs            = Int[],
)

n_total = length(OBJETOS) * length(T0_VALUES) * length(S1_EPOCHS) * length(EPS_CAUSAL_VALUES)
run_idx = 0

for obj in OBJETOS
    for t0_init in T0_VALUES
        for s1 in S1_EPOCHS
            for εc in EPS_CAUSAL_VALUES
                global run_idx, results
                run_idx += 1
                d = Dates.now()
                @printf("\n%s\n", repeat("=", 80))
                @printf("🔬 Grid run %d/%d | %s | t0=%.4f | s1=%d | ε_causal=%.2f\n",
                        run_idx, n_total, obj.nombre, t0_init, s1, εc)
                @printf("%s\n", repeat("=", 80))
                flush(stdout)

                r = run_pinn_experiment(
                    archivo       = obj.archivo,
                    z             = obj.z,
                    t0_init       = t0_init,
                    lit_ni_range  = obj.lit_ni,
                    s1_epochs     = s1,
                    s2_epochs     = S2_EPOCHS,
                    ε_causal      = εc,
                    output_prefix = "grid_$(obj.nombre)_t0$(t0_init)_s1$(s1)_ec$(εc)",
                )

                push!(results, (
                    obj.nombre,
                    obj.z,
                    r.t0_inicial,
                    s1,
                    S2_EPOCHS,
                    r.ε_causal,
                    r.t0_final,
                    r.t0_final_dias,
                    r.M_Ni0_final,
                    r.lit_ni,
                    r.κ_promedio,
                    r.rise_time_pinn,
                    r.loss_final,
                    r.λ_final,
                    r.T_MAX,
                    r.n_obs,
                ))

                CSV.write("resultados_grid.csv", results)
                @printf("✅ Guardado incremental. Total acumulado: %d filas\n", nrow(results))
                flush(stdout)
            end
        end
    end
end

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

println("\n" * "=" ^ 80)
println("🏁 Grid Search Completado — Resultados Consolidados")
println("=" ^ 80)
println("Archivo: resultados_grid.csv")
println("Filas: $(nrow(results))")
println()
sort!(results, [:objeto, :t0_inicial, :s1_epochs])
show(results, allcols=true)
println()
