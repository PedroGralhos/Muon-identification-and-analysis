
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pasta = Path(r"C:\Users\Utilizador\Desktop\Estágio LIP")
df = pd.read_parquet(pasta / "clusters_muoes_completos_background_wPb.parquet")

print("Available columns:", df.columns.tolist())
print(f"Total saved rows/pixels: {len(df)}")

lista_de_dataframes = []

zi = 0.0
zf = 2.0

for (ficheiro, cluster_id), df_muao_isolado in df.groupby(["ficheiro_origem", "Cluster"]):
    
    print(f"\nAnalyzing Cluster {cluster_id} from file {ficheiro}:")
    print(f"This muon has {len(df_muao_isolado)} activated pixels.")

    df_muao = df_muao_isolado.copy().sort_values(by="Ns")

    x = df_muao["X"].to_numpy()
    y = df_muao["Y"].to_numpy()
    ns = df_muao["Ns"].to_numpy()

    t_ref_min = ns.min()
    t_max_abs = ns.max()
    tempo_deriva_relativo = ns - t_ref_min
    dt_max_relativo = tempo_deriva_relativo.max()

    n_pixeis = len(df_muao)
    if n_pixeis > 1:
        zk_array = np.linspace(zi, zf, n_pixeis)
    else:
        zk_array = np.array([zi])

    df_muao["tempo_deriva_relativo_ns"] = tempo_deriva_relativo
    df_muao["t_min_ref_ns"] = t_ref_min
    df_muao["t_max_ref_ns"] = t_max_abs
    df_muao["delta_tempo_cluster_ns"] = dt_max_relativo
    df_muao["Z_mm"] = zk_array

    df_muao["xmin"], df_muao["xmax"] = x.min(), x.max()
    df_muao["ymin"], df_muao["ymax"] = y.min(), y.max()
    df_muao["n_pixeis_cluster"] = n_pixeis

    lista_de_dataframes.append(df_muao)

if lista_de_dataframes:
    df_todos_muoes = pd.concat(lista_de_dataframes, ignore_index=True)
    ficheiro_saida = pasta / "lista_muoes_completa_background_wPb.parquet"
    df_todos_muoes.to_parquet(ficheiro_saida, index=False)
    
    print(f"\nAll {len(lista_de_dataframes)} muons were saved to:")
    print(f" -> {ficheiro_saida}")

    t_global = df_todos_muoes["tempo_deriva_relativo_ns"].to_numpy()
    z_global = df_todos_muoes["Z_mm"].to_numpy()

    coluna_chip = "Overflow"
    
    # Global Raw Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_global, z_global, 'o', color='tab:red', alpha=0.6, label='Pixels (All Clusters)', markersize=5)

    if len(t_global) > 1:
        coeffs, cov = np.polyfit(t_global, z_global, 1, cov=True)
        m, b = coeffs
        sigma_m, sigma_b = np.sqrt(np.diag(cov))
        
        t_fit = np.linspace(t_global.min(), t_global.max(), 100)
        z_fit = m * t_fit + b
        
        ax.plot(t_fit, z_fit, '--', color='navy', label='Global Linear Fit', linewidth=2)

        sinal = "+" if b >= 0 else "-"
        equacao_texto = (f"$Z(t) = ({m:.4f} \\pm {sigma_m:.4f}) \\cdot \\Delta t$\n"
                         f"${sinal} ({abs(b):.4f} \\pm {sigma_b:.4f})$")
        ax.legend(title=equacao_texto, loc='upper left', fontsize=10, title_fontsize=10)
    else:
        ax.legend(loc='upper left')

    ax.set_xlabel("Relative Drift Time $\Delta t$ (ns)", fontsize=11)
    ax.set_ylabel("Depth $Z$ (mm)", fontsize=11)
    ax.set_title("Global Depth Profile $Z(t)$ — All Muons (Raw)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(zi - 0.1, zf + 0.1)
    plt.tight_layout()
    plt.show()

    chips = []
    if coluna_chip in df_todos_muoes.columns:
        chips = sorted(df_todos_muoes[coluna_chip].unique())
        print("\n--- Generating individual raw plots per Chip ---")
        
        for chip in chips:
            df_chip = df_todos_muoes[df_todos_muoes[coluna_chip] == chip]
            t_chip = df_chip["tempo_deriva_relativo_ns"].to_numpy()
            z_chip = df_chip["Z_mm"].to_numpy()

            fig_chip, ax_chip = plt.subplots(figsize=(8, 6))
            ax_chip.plot(t_chip, z_chip, 'o', color='tab:green', alpha=0.6,
                         label=f'Pixels (Chip {chip})', markersize=5)

            if len(t_chip) > 1:
                coeffs_chip, cov_chip = np.polyfit(t_chip, z_chip, 1, cov=True)
                m_chip, b_chip = coeffs_chip
                sigma_m_chip, sigma_b_chip = np.sqrt(np.diag(cov_chip))

                t_fit_chip = np.linspace(t_chip.min(), t_chip.max(), 100)
                z_fit_chip = m_chip * t_fit_chip + b_chip

                ax_chip.plot(t_fit_chip, z_fit_chip, '--', color='black', label=f'Chip {chip} Fit', linewidth=2)

                sinal_c = "+" if b_chip >= 0 else "-"
                equacao_texto_c = (f"$Z(t) = ({m_chip:.4f} \\pm {sigma_m_chip:.4f}) \\cdot \\Delta t$\n"
                                   f"${sinal_c} ({abs(b_chip):.4f} \\pm {sigma_b_chip:.4f})$")
                ax_chip.legend(title=equacao_texto_c, loc='upper left', fontsize=10, title_fontsize=10)
            else:
                ax_chip.legend(loc='upper left')

            ax_chip.set_xlabel("Relative Drift Time $\Delta t$ (ns)", fontsize=11)
            ax_chip.set_ylabel("Depth $Z$ (mm)", fontsize=11)
            ax_chip.set_title(f"Raw Depth Profile $Z(t)$ — Chip {chip}", fontsize=12)
            ax_chip.grid(True, linestyle='--', alpha=0.6)
            ax_chip.set_ylim(zi - 0.1, zf + 0.1)
            plt.tight_layout()
            plt.show()

    # Asking for Z limits
    while True:
        try:
            z_min_janela = float(input(f"\nLower Z limit (mm) [between {zi} and {zf}]: ").strip().replace(",", "."))
            z_max_janela = float(input(f"Upper Z limit (mm) [between {zi} and {zf}]: ").strip().replace(",", "."))

            if z_min_janela >= z_max_janela or z_min_janela < zi or z_max_janela > zf:
                print(f"Invalid limits. They must be within [{zi}, {zf}]. Try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number (e.g., 0.2).")

    # Asking for sigmas for Chip 1
    while True:
        try:
            n_sigmas_chip1 = float(input("\nHow many standard deviations (\u03c3) do you want to accept for CHIP 1? ").strip().replace(",", "."))
            if n_sigmas_chip1 <= 0:
                print("The number of \u03c3 must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Asking for sigmas for Chip 2
    while True:
        try:
            n_sigmas_chip2 = float(input("\nHow many standard deviations (\u03c3) do you want to accept for CHIP 2? ").strip().replace(",", "."))
            if n_sigmas_chip2 <= 0:
                print("The number of \u03c3 must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    velocidades = []
    incertezas = []
    
    t_global_limpo_list = []
    z_global_limpo_list = []

    print(f"\n--- Applying window to all and N*\u03c3 filter to Chips 1 ({n_sigmas_chip1}\u03c3) and 2 ({n_sigmas_chip2}\u03c3) ---")

    # Dictionary mapping chips to their respective cutoff limits
    sigmas_por_chip = {1: n_sigmas_chip1, 2: n_sigmas_chip2}

    if coluna_chip in df_todos_muoes.columns and len(chips) > 0:
        for chip in chips:
            df_chip = df_todos_muoes[df_todos_muoes[coluna_chip] == chip]
            t_chip = df_chip["tempo_deriva_relativo_ns"].to_numpy()
            z_chip = df_chip["Z_mm"].to_numpy()

            # Z Window
            mascara_j = (z_chip >= z_min_janela) & (z_chip <= z_max_janela)
            t_chip, z_chip = t_chip[mascara_j], z_chip[mascara_j]

            # Iterative filtering for defined chips (1 and 2)
            if chip in sigmas_por_chip and len(t_chip) > 1:
                n_sigmas_atual = sigmas_por_chip[chip]
                mascara_c_iter = np.ones(len(t_chip), dtype=bool)
                n_iteracoes_max = 5
                
                for _ in range(n_iteracoes_max):
                    t_c_temp = t_chip[mascara_c_iter]
                    z_c_temp = z_chip[mascara_c_iter]
                    
                    if len(t_c_temp) <= 1:
                        break
                        
                    coeffs_c = np.polyfit(t_c_temp, z_c_temp, 1)
                    z_pred_c = coeffs_c[0] * t_chip + coeffs_c[1]
                    
                    residuos_c = z_chip - z_pred_c
                    std_c = np.std(residuos_c[mascara_c_iter])
                    
                    nova_mascara_c = np.abs(residuos_c) <= (n_sigmas_atual * std_c)
                    
                    if np.array_equal(mascara_c_iter, nova_mascara_c):
                        break
                    mascara_c_iter = nova_mascara_c

                t_chip_limpo = t_chip[mascara_c_iter]
                z_chip_limpo = z_chip[mascara_c_iter]
                print(f"  Chip {chip} (Filtered at {n_sigmas_atual}\u03c3): kept {len(t_chip_limpo)} out of {len(t_chip)} points.")
            else:
                t_chip_limpo, z_chip_limpo = t_chip, z_chip
                print(f"  Chip {chip}: kept {len(t_chip_limpo)} points (Z Window only).")

            t_global_limpo_list.append(t_chip_limpo)
            z_global_limpo_list.append(z_chip_limpo)

            if len(t_chip_limpo) > 1:
                coeffs_chip, cov_chip = np.polyfit(t_chip_limpo, z_chip_limpo, 1, cov=True)
                m_chip = coeffs_chip[0]
                sigma_m_chip = np.sqrt(cov_chip[0, 0])
            else:
                m_chip, sigma_m_chip = np.nan, np.nan

            velocidades.append(m_chip)
            incertezas.append(sigma_m_chip)
            print(f"v_chip_{chip} = {m_chip:.4f} ± {sigma_m_chip:.4f} mm/ns")

    t_global_limpo = np.concatenate(t_global_limpo_list)
    z_global_limpo = np.concatenate(z_global_limpo_list)

    # Clean Global Plot
    fig, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(t_global_limpo, z_global_limpo, 'o', color='tab:red', alpha=0.6,
             label=f'Clean Pixels (Chip 1: {n_sigmas_chip1}\u03c3 | Chip 2: {n_sigmas_chip2}\u03c3)', markersize=5)

    if len(t_global_limpo) > 1:
        coeffs_filt, cov_filt = np.polyfit(t_global_limpo, z_global_limpo, 1, cov=True)
        m_filt, b_filt = coeffs_filt
        sigma_m_filt, sigma_b_filt = np.sqrt(np.diag(cov_filt))

        t_fit_filt = np.linspace(t_global_limpo.min(), t_global_limpo.max(), 100)
        z_fit_filt = m_filt * t_fit_filt + b_filt

        ax3.plot(t_fit_filt, z_fit_filt, '--', color='navy', label='Clean Global Linear Fit', linewidth=2)

        sinal_filt = "+" if b_filt >= 0 else "-"
        equacao_texto_filt = (f"$Z(t) = ({m_filt:.4f} \\pm {sigma_m_filt:.4f}) \\cdot \\Delta t$\n"
                              f"${sinal_filt} ({abs(b_filt):.4f} \\pm {sigma_b_filt:.4f})$")
        ax3.legend(title=equacao_texto_filt, loc='upper left', fontsize=10, title_fontsize=10)
    else:
        ax3.legend(loc='upper left')

    ax3.set_xlabel("Relative Drift Time $\Delta t$ (ns)", fontsize=11)
    ax3.set_ylabel("Depth $Z$ (mm)", fontsize=11)
    ax3.set_title(f"Clean Global Profile — Window {z_min_janela}–{z_max_janela} mm", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(zi - 0.1, zf + 0.1)
    plt.tight_layout()
    plt.show()

    # Velocities Plot
    fig, ax4 = plt.subplots(figsize=(7, 5))
    ax4.errorbar(chips, velocidades, yerr=incertezas, fmt='o', color='tab:blue',
                 capsize=5, markersize=8, linewidth=1.5, ecolor='tab:blue')

    ax4.set_xlabel("Chip / Detector", fontsize=11)
    ax4.set_ylabel("Drift Velocity (mm/ns)", fontsize=11)
    ax4.set_title("Drift Velocity per Chip (Filtered)", fontsize=12)
    ax4.set_xticks(chips)
    ax4.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

else:
    print("\nThe DataFrame list is empty. No files were saved.")