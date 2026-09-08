from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pasta = Path(r"C:\Users\Utilizador\Desktop\Estágio LIP")

df = pd.read_parquet(pasta / "clusters_muoes_completos_co57.parquet")

print("Colunas disponíveis:", df.columns.tolist())
print(f"Total de linhas/pixeis guardados: {len(df)}")


lista_de_dataframes = []

for (ficheiro, cluster_id), df_muon_isolado in df.groupby(["ficheiro_origem", "Cluster"]):
    
    print(f"\nA analisar o Cluster {cluster_id} vindo do ficheiro {ficheiro}:")
    print(f"Este muão tem {len(df_muon_isolado)} pixeis ativados.")

    x = df_muon_isolado["X"].to_numpy()
    y = df_muon_isolado["Y"].to_numpy()
    tot = df_muon_isolado["ToT (keV)"].to_numpy()
    ns = df_muon_isolado["Ns"].to_numpy()

    ovf_val = df_muon_isolado["Overflow"].iloc[0]
    r_sq = df_muon_isolado["muon_r_squared"].iloc[0]
    d_mm = df_muon_isolado["muon_distancia_d_mm"].iloc[0]
    energia_cluster = df_muon_isolado["muon_energia_total_keV"].iloc[0]
    stopping_power = df_muon_isolado["muon_stopping_power_keV_mm"].iloc[0]

    idx_tmin = np.argmin(ns)
    t_ref_min = ns[idx_tmin]     
    y_tmin = y[idx_tmin]         

    tempo_deriva_relativo = ns - t_ref_min

    idx_tmax = np.argmax(ns)
    t_max_abs = ns[idx_tmax]
    dt_max_relativo = tempo_deriva_relativo[idx_tmax]
    y_tmax = y[idx_tmax]         

    delta_y_px = abs(y_tmax - y_tmin)
    delta_y_mm = delta_y_px * 0.055 


    df_muon = df_muon_isolado.copy()
    

    df_muon["tempo_deriva_relativo_ns"] = tempo_deriva_relativo
    df_muon["t_min_ref_ns"] = t_ref_min
    df_muon["t_max_ref_ns"] = t_max_abs
    df_muon["delta_tempo_cluster_ns"] = dt_max_relativo
    

    df_muon["delta_y_mm"] = delta_y_mm
    df_muon["xmin"] = x.min()
    df_muon["xmax"] = x.max()
    df_muon["ymin"] = y.min()
    df_muon["ymax"] = y.max()
    df_muon["n_pixeis_cluster"] = len(df_muon_isolado)


    lista_de_dataframes.append(df_muon)

    max_x = int(x.max()) + 1
    max_y = int(y.max()) + 1
    
    heatmap_deriva = np.full((max_y, max_x), np.nan)
    heatmap_deriva[y.astype(int), x.astype(int)] = tempo_deriva_relativo
    
    x_esq, x_dir = x.min(), x.max()
    y_baixo, y_cima = y.min(), y.max()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heatmap_deriva, origin="lower", cmap="magma")
    
    ax.set_xlim(x_esq - 1, x_dir + 1)
    ax.set_ylim(y_baixo - 1, y_cima + 1)
    
    ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    nome_ficheiro_sem_ext = Path(ficheiro).stem
    
    ax.set_title(f"{nome_ficheiro_sem_ext} | CdTe - Cluster {cluster_id}\n", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\Delta t$ em relação ao $t_{min}$ (ns)", rotation=270, labelpad=15)
    
    ax.set_xlabel("X (pixeis)")
    ax.set_ylabel("Y (pixeis)")
    
    plt.tight_layout()
    plt.show()

print(f"\nTotal de muões/DataFrames na lista: {len(lista_de_dataframes)}")


if lista_de_dataframes:
    df_todos_muoes = pd.concat(lista_de_dataframes, ignore_index=True)
    
    ficheiro_saida = pasta / "lista_muoes_completa_co57.parquet"
    df_todos_muoes.to_parquet(ficheiro_saida, index=False)
    
    print(f"\n Todos os {len(lista_de_dataframes)} muões foram guardados em:")
    print(f" -> {ficheiro_saida}")
else:
    print("\nA lista de DataFrames está vazia. Nenhum ficheiro foi guardado.")