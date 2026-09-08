from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pasta = Path(r"C:\Users\Utilizador\Desktop\Estágio LIP")
df = pd.read_parquet(pasta / "clusters_muoes_completos_ba133.parquet")

print("Colunas disponíveis:", df.columns.tolist())
print(f"Total de linhas/pixeis guardados: {len(df)}")

lista_de_dataframes = []

zi = 0.0
zf = 2.0 


for (ficheiro, cluster_id), df_muao_isolado in df.groupby(["ficheiro_origem", "Cluster"]):
    
    print(f"\nA analisar o Cluster {cluster_id} vindo do ficheiro {ficheiro}:")
    print(f"Este muão tem {len(df_muao_isolado)} pixeis ativados.")


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


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))


    max_x, max_y = int(x.max()) + 1, int(y.max()) + 1
    heatmap_deriva = np.full((max_y, max_x), np.nan)
    heatmap_deriva[y.astype(int), x.astype(int)] = tempo_deriva_relativo

    im = ax1.imshow(heatmap_deriva, origin="lower", cmap="magma")
    ax1.set_xlim(x.min() - 1, x.max() + 1)
    ax1.set_ylim(y.min() - 1, y.max() + 1)
    ax1.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.set_title("Projeção no Plano XY (Pixeis)")
    ax1.set_xlabel("X (pixeis)")
    ax1.set_ylabel("Y (pixeis)")
    
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("$\Delta t$ em relação ao $t_{min}$ (ns)", rotation=270, labelpad=15)


    ax2.plot(tempo_deriva_relativo, zk_array, 'o', color='tab:red', label='Pixéis (Dados)', markersize=6)

    if n_pixeis > 1:
 
        m, b = np.polyfit(tempo_deriva_relativo, zk_array, 1)
        
        t_fit = np.linspace(tempo_deriva_relativo.min(), tempo_deriva_relativo.max(), 100)
        z_fit = m * t_fit + b
        
        ax2.plot(t_fit, z_fit, '--', color='navy', label='Ajuste Linear', linewidth=1.8)

        sinal = "+" if b >= 0 else "-"
        equacao_texto = f"$Z(t) = {m:.4f} \cdot \Delta t {sinal} {abs(b):.4f}$"
        
        ax2.legend(title=equacao_texto, loc='upper left', fontsize=9, title_fontsize=10)
    else:
        ax2.legend(loc='upper left')

    ax2.set_xlabel("Tempo de Deriva Relativo $\Delta t$ (ns)")
    ax2.set_ylabel("Profundidade $Z$ (mm)")
    ax2.set_title("Perfil de Profundidade $Z(t)$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(zi - 0.1, zf + 0.1)

    nome_ficheiro_sem_ext = Path(ficheiro).stem
    fig.suptitle(f"{nome_ficheiro_sem_ext} | CdTe - Cluster {cluster_id}", fontsize=12)

    plt.tight_layout()
    plt.show()


if lista_de_dataframes:
    df_todos_muoes = pd.concat(lista_de_dataframes, ignore_index=True)
    ficheiro_saida = pasta / "lista_muoes_completa_ba133.parquet"
    df_todos_muoes.to_parquet(ficheiro_saida, index=False)
    
    print(f"\nTodos os {len(lista_de_dataframes)} muões foram guardados em:")
    print(f" -> {ficheiro_saida}")
else:
    print("\nA lista de DataFrames está vazia. Nenhum ficheiro foi guardado.")
