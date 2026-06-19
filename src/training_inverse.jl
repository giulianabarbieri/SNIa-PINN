#!/usr/bin/env julia

"""
================================================================================
PINNs Modo 2: Inverso con Restricción Fuerte de CI, Resampleo e Importance Sampling
================================================================================

Adaptación del modelo de Difusividad D(x) a Opacidad κ(t) para SNIa.

Dados los datos observados [F_obs(t_i)], se busca **recuperar** κ = κ(t):
la opacidad como función del tiempo.

Arquitectura de dos redes con parámetros independientes β = [β_1, β_2]:
- Red β_1 (Solucionadora): Aproxima la curva de luz continua F_β1(t, fid).
- Red β_2 (Física/Opacidad): Aproxima la opacidad dinámica κ_β2(t) -> ℝ
                              (salida escalar, solo depende del tiempo).

Nuevas funcionalidades implementadas (Clases 8-9-10 de DM2026):

1. RESTRICCIÓN FUERTE DE CI (Clase 9 — Hard PINNs):
   F_θ(t) = φ(t - t₀) · NN(t) con φ(0) = 0
   t₀ (t0_explosion) es un parámetro aprendible que representa el día
   de explosión en coordenadas normalizadas. Cada SN aprende su propio t₀.

2. RESAMPLEO POR ÉPOCA (Clase 10 — Resampleo uniforme por época):
   Los puntos de colocación se renuevan periódicamente cada
   RESAMPLE_INTERVAL épocas para evitar overfitting a la grilla fija.

3. IMPORTANCE SAMPLING (Clase 10 — Importance Sampling):
   P(x_i^{k+1}) ∝ exp(α · |Residuo_i|) concentra puntos de colocación
   en regiones donde el residuo físico es mayor.

Función de Costo Total:
  L_total = L_empírica (Datos) + λ * L_física (Residuo EDO de Arnett con κ_β2(t))

donde:
  L_emp   = (1/N) Σ_i || F_θ(t_i) - F_obs(t_i) ||² / σ_i²   (con hard constraint)
  L_fís   = (1/M) Σ_j || dF_θ/dt(t_j) - Arnett(F_θ(t_j), κ_β2(t_j), M_Ni0) ||²
  t_since_expl = (t_norm - t₀) · T_MAX                       (tiempo físico real)

Operaciones clave:
  ∂ₜ F_θ   →  calculada vía diferencias finitas centradas sobre F_θ (con hard constraint)
  κ(t)     →  salida de β_2 con Softplus para garantizar κ > 0
  t₀       →  parámetro libre en β = [β_1, β_2, M_Ni0, t₀]
================================================================================
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Plots
using Lux
using Random
using ComponentArrays
using Zygote
using Optimisers
using Statistics

# ==============================================================================
# CONSTANTES FÍSICAS GLOBALES (no cambian entre corridas)
# ==============================================================================
const τ_Ni  = 8.44f0          # Vida media del Ni-56 [días]
const τ_Co  = 111.26f0        # Vida media del Co-56 [días]

# Constantes de tasa energética: convertidas de erg/s/g a erg/s/M_sol
# multiplicando por M☉ = 1.989e33 g/M_sol.
# Usamos Float64 porque ϵ_Ni0 ≈ 7.76e43 > Float32 max (≈ 3.4e38).
const ϵ_Ni0 = 3.90e10 * 1.989e33  # ≈ 7.76e43 erg/s/M_sol (Float64)
const ϵ_Co0 = 6.78e9 * 1.989e33   # ≈ 1.35e43 erg/s/M_sol (Float64)

# ── Configuración de puntos de colocación y resampleo ────────────────────────
const N_COL_DEFAULT    = 120   # Cantidad base de puntos de colocación
const RESAMPLE_INTERVAL = 100  # Re-muestrear cada N épocas
const ALPHA_IS         = 2.0f0 # Agresividad del importance sampling

# ==============================================================================
# FUNCIONES FÍSICAS (independientes del dataset)
# ==============================================================================

"""
    arnett_source(t_real, M_Ni0_pred)

Calcula la tasa de producción energética normalizada de la desintegración
radiactiva Ni⁵⁶ → Co⁵⁶ → Fe⁵⁶ en el modelo de Arnett.

Retorna: (ε_Ni(t) + ε_Co(t)) / LUMINOSITY_MAX  (adimensional)

donde:
    ε_Ni(t) = ϵ_Ni0 · M_Ni0 · exp(-t/τ_Ni)
    ε_Co(t) = ϵ_Co0 · M_Ni0 · [exp(-t/τ_Co) - exp(-t/τ_Ni)] / (1 - τ_Ni/τ_Co)

La normalización por LUMINOSITY_MAX permite que el término fuente sea
comparable con el flujo normalizado F(t) en la EDO de Arnett.

Nota: ϵ_Ni0 y ϵ_Co0 están en erg/s/M_sol (convertidas desde erg/s/g
multiplicando por M☉ = 1.989e33 g/M_sol).
"""
function arnett_source(t_real, M_Ni0_pred, L_MAX)
    t_eval = max(0.0f0, t_real)
    ε_Ni = Float64(3.90e10 * 1.989e33) * Float64(M_Ni0_pred) *
           exp(-Float64(t_eval) / Float64(τ_Ni))
    ε_Co = Float64(6.78e9 * 1.989e33) * Float64(M_Ni0_pred) *
           (exp(-Float64(t_eval) / Float64(τ_Co)) - exp(-Float64(t_eval) / Float64(τ_Ni))) /
           (1.0 - Float64(τ_Ni) / Float64(τ_Co))
    return Float32((ε_Ni + ε_Co) / Float64(L_MAX))
end

"""
    arnett_residual_dynamic(t_real, F_current, dF_dt_real, κ_t, M_Ni0_pred, L_MAX)

Calcula el residuo de la EDO de Arnett con unidades consistentes:

    Residuo = dF/dt - κ · (ε/L_max - F)

donde:
- dF/dt : derivada temporal de F (unidades: 1/día)
- κ     : opacidad efectiva (unidades: 1/día)
- ε     : tasa de producción energética (erg/s)
- L_max : luminosidad máxima observada (erg/s)
- F     : flujo normalizado (adimensional)

La forma correcta de la ODE es  dF/dt = κ · (ε/L_max - F),
donde κ provee la escala temporal (1/día). El término fuente ε/L_max
es adimensional y κ le da las unidades correctas.

Comparar con Clase 9: la ecuación de Arnett relaciona la derivada temporal
de la luminosidad con la fuente de energía radiactiva y la opacidad.
"""
function arnett_residual_dynamic(t_real, F_current, dF_dt_real, κ_t, M_Ni0_pred, L_MAX)
    fuente_norm = arnett_source(t_real, M_Ni0_pred, L_MAX)  # ε/L_max (adimensional)
    edo_rhs = κ_t * (fuente_norm - F_current)
    return dF_dt_real - edo_rhs
end

# ==============================================================================
# FUNCIONES DE RESTRICCIÓN Y MUESTREO
# ==============================================================================

"""
    apply_hard_constraint(t_norm, raw_flux, t0_val)

Aplica la restricción fuerte de explosión (F(t₀) = 0) mediante reparametrización:

    F_física(t) = φ(t - t₀) · NN(t)

donde φ(τ) = τ para τ > 0, y 0 en caso contrario.

Referencia: Clase 9 — "Hard PINNs" y reparametrización u_θ(t) = u₀ + φ(t)·NN_θ(t).
"""
function apply_hard_constraint(t_norm, raw_flux, t0_val)
    τ = t_norm - t0_val
    return ifelse(τ > 0.0f0, τ * raw_flux, 0.0f0)
end

"""
    resample_physics_points!(t_physics)

Re-genera los puntos de colocación para evitar overfitting a una grilla fija
(Clase 10 — Resampleo uniforme por época).
"""
function resample_physics_points!(t_physics; method=:hybrid)
    if method == :hybrid
        u = range(0.0f0, 1.0f0, length=N_COL_DEFAULT)
        t_physics .= Float32.((exp.(2.0f0 .* u) .- 1.0f0) / (exp(2.0f0) - 1.0f0))
    elseif method == :uniform
        t_physics .= Float32.(sort(rand(Float32, N_COL_DEFAULT)))
    end
    return nothing
end

"""
    importance_sample!(t_physics, residuals_per_point, α=ALPHA_IS)

Genera nuevos puntos de colocación mediante Importance Sampling (Clase 10).

    P(x_i^{k+1}) ∝ exp(α · |residual_i|)

Concentra poder de cómputo en las regiones donde la PINN está errando más.
"""
function importance_sample!(t_physics, residuals_per_point::Vector{Float32}, α::Float32=ALPHA_IS)
    n_phys  = length(t_physics)
    n_total = length(residuals_per_point)

    n_points = div(n_total, 2)
    point_residuals = Float32[
        sqrt(residuals_per_point[2*i-1]^2 + residuals_per_point[2*i]^2)
        for i in 1:n_points
    ]

    max_res = maximum(point_residuals)
    weights = exp.(α .* (point_residuals .- max_res))
    weights ./= sum(weights)

    cdf = cumsum(weights)
    cdf[end] = 1.0f0
    u = sort!(rand(Float32, n_points))
    idx = zeros(Int, n_points)
    j = 1
    for i in 1:n_points
        while cdf[j] < u[i]
            j += 1
        end
        idx[i] = j
    end
    t_physics .= t_physics[idx]
    return nothing
end

"""
    causal_importance_sample!(t_physics, residuals_per_point, α=ALPHA_IS, ε_causal=1.0f0)

Entrenamiento Causal + Importance Sampling combinados.

Referencia: Wang, Sankaran & Perdikaris (2022), "Respecting causality is all
you need for training PINNs", CMAME.

La idea: en lugar de entrenar la EDO en todo el dominio temporal simultáneamente,
se enfoca el entrenamiento en tiempos tempranos primero (cerca de la explosión,
donde el source de Arnett es fuerte y la señal física es clara), y se expande
progresivamente a tiempos posteriores a medida que los residuos tempranos bajan.

Mecanismo: pesos causales  W(t_i) = exp(-ε_causal · Σ_{k=1}^{i-1} Residuo²(t_k))
donde los puntos se ordenan por tiempo. Si los residuos tempranos son altos,
los pesos para tiempos posteriores se suprimen exponencialmente, forzando al
optimizador a resolver la física en orden cronológico.

Estos pesos se multiplican con los del importance sampling:
  P(x_i) ∝ exp(α · |Residuo_i|) · W(t_i)

Combinando ambas estrategias: concentramos puntos donde el error es mayor
(importance sampling) pero respetando la estructura causal de la EDO.
"""
function causal_importance_sample!(t_physics, residuals_per_point::Vector{Float32};
                                   α::Float32=ALPHA_IS, ε_causal::Float32=1.0f0)
    n_phys  = length(t_physics)
    n_total = length(residuals_per_point)
    n_points = div(n_total, 2)

    # Residuos por punto (combinando bandas g y r)
    point_residuals = Float32[
        sqrt(residuals_per_point[2*i-1]^2 + residuals_per_point[2*i]^2)
        for i in 1:n_points
    ]

    # Pesos del importance sampling: P ∝ exp(α · |res|)
    max_res = maximum(point_residuals)
    is_weights = exp.(α .* (point_residuals .- max_res))

    # Pesos causales (Wang et al., 2022):
    #   W(t_i) = exp(-ε_causal · Σ_{k=1}^{i-1} res²_k)
    # Los puntos ya están ordenados por t_physics (creciente en tiempo)
    accumulated = 0.0f0
    causal_weights = similar(is_weights)
    for i in 1:n_points
        # El peso causal del punto i depende de los residuos ACUMULADOS hasta i-1
        causal_weights[i] = exp(-ε_causal * accumulated)
        # Acumulamos el residuo del punto i para el siguiente
        accumulated += point_residuals[i]^2
    end

    # Pesos combinados: importance sampling × pesos causales
    weights = is_weights .* causal_weights
    weights ./= sum(weights)

    cdf = cumsum(weights)
    cdf[end] = 1.0f0
    u = sort!(rand(Float32, n_points))
    idx = zeros(Int, n_points)
    j = 1
    for i in 1:n_points
        while cdf[j] < u[i]
            j += 1
        end
        idx[i] = j
    end
    t_physics .= t_physics[idx]
    return nothing
end

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

"""
    run_pinn_experiment(; archivo, z, t0_init, lit_ni_range, s1_epochs, s2_epochs,
                         output_prefix="pinn_inversa_modo2")

Ejecuta el pipeline completo de PINN Inversa Modo 2 para un objeto ZTF.

Parámetros:
- `archivo`: ruta al CSV de detecciones (ej. "dataset/ZTF25aavdmzf_detections.csv")
- `z`: redshift cosmológico del objeto
- `t0_init`: valor inicial de t0_explosion en t_norm (ej. 0.0f0, -0.1f0)
- `lit_ni_range`: rango de M_Ni0 de literatura como string (ej. "0.52 – 0.58")
- `s1_epochs`: épocas de Stage 1 (solo datos)
- `s2_epochs`: épocas de Stage 2 (datos + física)
- `ε_causal`: factor de agresividad del entrenamiento causal
  (Wang, Sankaran & Perdikaris, 2022). ε alto → más restrictivo.
- `output_prefix`: prefijo para archivos de salida gráfica

 Retorna un NamedTuple con las métricas finales:
  (t0_inicial, t0_final, t0_final_dias, M_Ni0_final, κ_promedio,
   rise_time_pinn, loss_final, ε_causal)
"""
function run_pinn_experiment(;
    archivo::String,
    z::Float64,
    t0_init::Float32,
    lit_ni_range::String,
    s1_epochs::Int,
    s2_epochs::Int,
    ε_causal::Float32 = 0.3f0,
    output_prefix::String = "pinn_inversa_modo2",
)

    # ────────────────────────────────────────────────────────────────────────
    # 1. CARGA Y PREPROCESAMIENTO COSMOLÓGICO
    # ────────────────────────────────────────────────────────────────────────

    # Constantes cosmológicas
    H_0     = 70.0               # km/s/Mpc
    c_light = 299792.458         # km/s
    D_Mpc   = (c_light * z) / H_0
    D_cm    = D_Mpc * 3.085677581e24  # conversión Mpc → cm (CGS)

    # Carga de datos
    df_raw = CSV.read(archivo, DataFrame)
    df_clean = filter(row -> row.dubious == false && !isnan(row.sigmapsf), df_raw)
    df_clean = filter(row -> !isnan(row.magpsf) && row.magpsf > 0.0, df_clean)

    # Conversión de magnitud a luminosidad
    function mag_to_luminosity(mag)
        if mag <= 0.0 || isnan(mag)
            return 0.0
        end
        mag_abs = mag - 5.0 * log10(D_Mpc) - 25.0
        exponente = min(-0.4 * mag_abs, 50.0)
        return 3.0128e35 * (10.0^exponente)
    end

    propagate_error(mag, sigmapsf, flux) = flux * log(10.0f0) * 0.4f0 * sigmapsf

    # Construcción del DataFrame limpio
    df_objeto = DataFrame()
    t_min = minimum(df_clean.mjd)
    df_objeto.t          = Float32.(df_clean.mjd .- t_min)
    df_objeto.fid        = Float32.(df_clean.fid)
    luminosidades        = mag_to_luminosity.(df_clean.magpsf)
    df_objeto.flux       = luminosidades
    df_objeto.error_flux = propagate_error.(df_clean.magpsf, df_clean.sigmapsf, df_objeto.flux)
    sort!(df_objeto, :t)

    # Normalización
    T_MAX          = Float32(maximum(df_objeto.t))
    LUMINOSITY_MAX = maximum(filter(x -> !isinf(x) && !isnan(x), df_objeto.flux))

    safe_norm(v, M) = (isnan(v / M) || isinf(v / M)) ? 0.0f0 : Float32(v / M)

    flujos_normalizados  = [safe_norm(l, LUMINOSITY_MAX) for l in df_objeto.flux]
    errores_normalizados = [safe_norm(e, LUMINOSITY_MAX) for e in df_objeto.error_flux]
    errores_normalizados = [e == 0.0f0 ? 1.0f-3 : e for e in errores_normalizados]

    sn_data = DataFrame(
        t_norm     = Float32.(df_objeto.t ./ T_MAX),
        flux_norm  = Float32.(flujos_normalizados),
        error_norm = Float32.(errores_normalizados),
        fid        = df_objeto.fid
    )

    # ────────────────────────────────────────────────────────────────────────
    # 2. DEFINICIÓN DE LAS DOS REDES NEURONALES (β_1 y β_2)
    # ────────────────────────────────────────────────────────────────────────

    model_flux = Chain(
        Dense(2 => 32, tanh),
        Dense(32 => 32, tanh),
        Dense(32 => 16, tanh),
        Dense(16 => 1)
    )

    model_opacity = Chain(
        Dense(1 => 16, tanh),
        Dense(16 => 16, tanh),
        Dense(16 => 1, softplus)
    )

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    ps_flux,    st_flux    = Lux.setup(rng, model_flux)
    ps_opacity, st_opacity = Lux.setup(rng, model_opacity)

    # ────────────────────────────────────────────────────────────────────────
    # 3. PARÁMETROS UNIFICADOS
    # ────────────────────────────────────────────────────────────────────────

    ps_total = ComponentArray(
        beta1 = ps_flux,
        beta2 = ps_opacity,
        M_Ni0 = [0.2f0],
        t0_explosion = [t0_init]
    )

    T0_INICIAL = Float32(t0_init)

    # Puntos de colocación iniciales
    t_physics = let u = range(0.0f0, 1.0f0, length=N_COL_DEFAULT)
        Float32.((exp.(2.0f0 .* u) .- 1.0f0) / (exp(2.0f0) - 1.0f0))
    end

    # ────────────────────────────────────────────────────────────────────────
    # 4. FUNCIÓN DE LOSS
    # ────────────────────────────────────────────────────────────────────────

    function calculate_loss(ps, λ_current)
        ps_net1  = ps.beta1
        ps_net2  = ps.beta2
        M_Ni0_v  = ps.M_Ni0[1]
        t0_val   = ps.t0_explosion[1]
        L_MAX    = LUMINOSITY_MAX

        # Pérdida empírica
        loss_data = 0.0f0
        n_data    = nrow(sn_data)

        for i in 1:n_data
            inp          = [sn_data.t_norm[i]; sn_data.fid[i]]
            F_raw, _     = model_flux(inp, ps_net1, st_flux)
            F_pred       = apply_hard_constraint(sn_data.t_norm[i], F_raw[1], t0_val)
            χ            = (F_pred - sn_data.flux_norm[i]) / sn_data.error_norm[i]
            loss_data   += χ^2
        end
        loss_data /= Float32(n_data)

        if λ_current == 0.0f0
            return loss_data
        end

        # Pérdida física
        loss_physics = 0.0f0
        h_diff       = 0.001f0
        n_phys       = length(t_physics)

        for i in 1:n_phys
            tp = t_physics[i]
            t_since_expl = (tp - t0_val) * T_MAX

            κ_out, _ = model_opacity([tp], ps_net2, st_opacity)
            κ_t      = κ_out[1]

            for f in [1.0f0, 2.0f0]
                F_fut, _ = model_flux([tp + h_diff; f], ps_net1, st_flux)
                F_pas, _ = model_flux([tp - h_diff; f], ps_net1, st_flux)
                F_cur, _ = model_flux([tp; f], ps_net1, st_flux)

                F_fut_c = apply_hard_constraint(tp + h_diff, F_fut[1], t0_val)
                F_pas_c = apply_hard_constraint(tp - h_diff, F_pas[1], t0_val)
                F_cur_c = apply_hard_constraint(tp, F_cur[1], t0_val)

                dF_dt_real = (F_fut_c - F_pas_c) / (2.0f0 * h_diff * T_MAX)

                res = arnett_residual_dynamic(t_since_expl, F_cur_c, dF_dt_real, κ_t, M_Ni0_v, L_MAX)
                loss_physics += res^2
            end
        end
        loss_physics /= Float32(n_phys * 2)

        return loss_data + λ_current * loss_physics
    end

    function compute_physics_residuals(ps)
        ps_net1 = ps.beta1
        ps_net2 = ps.beta2
        M_Ni0_v = ps.M_Ni0[1]
        t0_val  = ps.t0_explosion[1]
        L_MAX   = LUMINOSITY_MAX
        h_diff  = 0.001f0

        residuals = Float32[]
        for tp in t_physics
            κ_out, _ = model_opacity([tp], ps_net2, st_opacity)
            κ_t = κ_out[1]
            t_since_expl = (tp - t0_val) * T_MAX

            for f in [1.0f0, 2.0f0]
                F_fut, _ = model_flux([tp + h_diff; f], ps_net1, st_flux)
                F_pas, _ = model_flux([tp - h_diff; f], ps_net1, st_flux)
                F_cur, _ = model_flux([tp; f], ps_net1, st_flux)

                F_fut_c = apply_hard_constraint(tp + h_diff, F_fut[1], t0_val)
                F_pas_c = apply_hard_constraint(tp - h_diff, F_pas[1], t0_val)
                F_cur_c = apply_hard_constraint(tp, F_cur[1], t0_val)

                dF_dt_real = (F_fut_c - F_pas_c) / (2.0f0 * h_diff * T_MAX)
                res = arnett_residual_dynamic(t_since_expl, F_cur_c, dF_dt_real, κ_t, M_Ni0_v, L_MAX)
                push!(residuals, Float32(res))
            end
        end
        return residuals
    end

    # ────────────────────────────────────────────────────────────────────────
    # 5. ENTRENAMIENTO EN DOS ETAPAS
    # ────────────────────────────────────────────────────────────────────────

    # STAGE 1: Solo datos
    opt1       = Optimisers.Adam(0.005f0)
    opt_state1 = Optimisers.setup(opt1, ps_total)

    for epoch in 1:s1_epochs
        current_loss, grads = Zygote.withgradient(p -> calculate_loss(p, 0.0f0), ps_total)

        # Congelamos t₀ en Stage 1
        grads[1].t0_explosion .= 0.0f0

        if epoch % RESAMPLE_INTERVAL == 0
            resample_physics_points!(t_physics; method=:hybrid)
        end
        opt_state1, ps_total = Optimisers.update(opt_state1, ps_total, grads[1])
    end

    # STAGE 2: Física activa con entrenamiento causal
    # Referencia: Wang, Sankaran & Perdikaris (2022), "Respecting
    #   causality is all you need for training PINNs", CMAME.
    opt2       = Optimisers.Adam(0.0005f0)
    opt_state2 = Optimisers.setup(opt2, ps_total)
    λ_phys     = 1.0f-5

    for epoch in 1:s2_epochs
        current_loss, grads = Zygote.withgradient(p -> calculate_loss(p, λ_phys), ps_total)

        if epoch % RESAMPLE_INTERVAL == 0
            residuals = compute_physics_residuals(ps_total)
            causal_importance_sample!(t_physics, residuals; α=ALPHA_IS, ε_causal=ε_causal)
        end
        opt_state2, ps_total = Optimisers.update(opt_state2, ps_total, grads[1])
    end

    loss_final = calculate_loss(ps_total, λ_phys)

    # ────────────────────────────────────────────────────────────────────────
    # 6. MÉTRICAS
    # ────────────────────────────────────────────────────────────────────────

    M_Ni0_final = ps_total.M_Ni0[1]
    t0_final    = ps_total.t0_explosion[1]
    t0_pinn_dias = -t0_final * T_MAX

    # κ en algunos puntos temporales
    κ_out_ref, _ = model_opacity([0.0f0], ps_total.beta2, st_opacity)
    κ0 = κ_out_ref[1]
    κ_out_ref, _ = model_opacity([0.5f0], ps_total.beta2, st_opacity)
    κ05 = κ_out_ref[1]
    κ_out_ref, _ = model_opacity([1.0f0], ps_total.beta2, st_opacity)
    κ1 = κ_out_ref[1]

    # κ promedio en una grilla fina
    t_grid = Float32.(range(0.0, 1.0, length=100))
    κ_vals = Float32[]
    for t_n in t_grid
        κ_out, _ = model_opacity([t_n], ps_total.beta2, st_opacity)
        push!(κ_vals, κ_out[1])
    end
    κ_prom = mean(κ_vals)

    # Pico observado
    peak_idx_data = argmax(sn_data.flux_norm)
    t_peak_norm   = sn_data.t_norm[peak_idx_data]
    t_peak_dias   = t_peak_norm * T_MAX
    rise_time_pinn = t_peak_dias + t0_pinn_dias

    return (
        t0_inicial       = Float32(T0_INICIAL),
        t0_final         = Float32(t0_final),
        t0_final_dias    = Float32(t0_pinn_dias),
        M_Ni0_final      = Float32(M_Ni0_final),
        κ_promedio       = Float32(κ_prom),
        κ0               = Float32(κ0),
        κ05              = Float32(κ05),
        κ1               = Float32(κ1),
        rise_time_pinn   = Float32(rise_time_pinn),
        loss_final       = Float32(loss_final),
        λ_final          = Float32(λ_phys),
        ε_causal         = ε_causal,
        lit_ni           = lit_ni_range,
        T_MAX            = Float32(T_MAX),
        L_MAX            = Float64(LUMINOSITY_MAX),
        n_obs            = nrow(sn_data),
        # Modelo entrenado para predicción externa
        ps_total         = ps_total,
        model_flux       = model_flux,
        st_flux          = st_flux,
    )

end

# ==============================================================================
# ENTRY POINT: si se ejecuta directamente
# ==============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_pinn_experiment(
        archivo       = "/home/kotyonok/Documents/SNIa-PINN/dataset/ZTF25aaxjntk_detections.csv",
        z             = 0.01633,
        t0_init       = -0.1f0,
        lit_ni_range  = "0.35 – 0.42",
        s1_epochs     = 2000,
        s2_epochs     = 3000,
        output_prefix = "pinn_inversa_modo2",
    )

    println("\n" * "=" ^ 80)
    println("🎉 ¡Entrenamiento PINN Modo 2 Completado con Éxito (Auditoría)!")
    println("=" ^ 80)
    println("  MÉTRICA FÍSICA          |   PINN MODO 2 (AI)   |   LITERATURA")
    println("-" ^ 80)
    println("  ☄️  Masa Níquel (M_sol)  |   $(rpad(round(result.M_Ni0_final; digits=4), 19)) |   $(rpad(result.lit_ni, 17))")
    println("  💥  Explosión t₀ (días) |   $(rpad(round(result.t0_final_dias; digits=2), 19)) |   —")
    println("  ⏱️  Rise Time (días)    |   $(rpad(round(result.rise_time_pinn; digits=2), 19)) |   18 (canónico)")
    println("-" ^ 80)
    println("  🛡️  κ promedio           = $(round(result.κ_promedio; digits=6))")
    println("  📍  t₀ inicial           = $(result.t0_inicial) (t_norm) ≈ $(-result.t0_inicial * result.T_MAX) días")
    println("=" ^ 80)
end
