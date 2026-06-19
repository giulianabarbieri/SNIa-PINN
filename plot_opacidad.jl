using Plots

"""
Script para parsear y graficar la evolución temporal de la opacidad κ(t)
generada por la PINN Inversa Modo 2 para el objeto ZTF25aaxjntk.
"""
function graficar_log_opacidad(ruta_archivo::String)
    dias_reales = Float64[]
    opacidades = Float64[]
    
    # 1. Lectura y parseo seguro del archivo de texto
    open(ruta_archivo, "r") do file
        for linea in eachline(file)
            # Ignoramos las líneas decorativas del log, encabezados y vacías
            if occursin("=", linea) || occursin("-", linea) || 
               occursin("PINN", linea) || occursin("Objeto", linea) || 
               occursin("Día", linea) || isempty(strip(linea))
                continue
            end
            
            # Dividimos la línea usando el carácter de la tubería '|'
            columnas = split(linea, "|")
            if length(columnas) >= 3
                try
                    # Convertimos las cadenas de texto a números flotantes
                    dia = parse(Float64, strip(columnas[1]))
                    kappa = parse(Float64, strip(columnas[3]))
                    
                    push!(dias_reales, dia)
                    push!(opacidades, kappa)
                catch
                    # Si alguna línea tiene un carácter extraño, la salteamos
                    continue
                end
            end
        end
    end
    
    # 2. Configuración estética del gráfico al estilo astrofísico
    plot(dias_reales, opacidades, 
         lw = 3, 
         color = :crimson, 
         label = "κ(t) recuperada (Red β₂)",
         xlabel = "Tiempo desde la explosión [Días reales]",
         ylabel = "Opacidad efectiva bolométrica κ [cm²/g]",
         title = "Evolución de la Opacidad en ZTF25aaxjntk (SN 2025oxy)",
         titlefontsize = 11,
         grid = true,
         gridalpha = 0.3,
         marker = (:circle, 3, :crimson),
         legend = :topleft,
         size = (700, 450))
         
    # Dibujamos una sutil línea vertical en el mínimo para marcar el cambio de régimen
    vline!([12.0], lw=1, linestyle=:dash, color=:gray, label="Mínimo (Día ~12)")
    
    # 3. Guardado obligatorio del archivo (sin usar show)
    output_name = "evolucion_opacidad_ztf25aaxjntk.png"
    savefig(output_name)
    println("📊 ¡Gráfico generado con éxito! Archivo guardado como: ", output_name)
end

# Ejecutamos el pipeline apuntando a tu archivo de log
# Asegurate de que el archivo 'opacidad_log.txt' esté en el mismo directorio o pasa la ruta correcta
graficar_log_opacidad("opacidad_log.txt")