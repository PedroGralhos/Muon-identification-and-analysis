from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress

pasta = Path(r"C:\Users\Utilizador\Desktop\Estágio LIP\ba133")

muoes_cand = []
dfs_muoes_confirmados = []

def extrair_numero(path):
    numeros = re.findall(r'\d+', path.stem)
    return int(numeros[-1]) if numeros else 0

todos_ficheiros = sorted(pasta.glob("*.parquet"), key=extrair_numero)

for ficheiro in todos_ficheiros:
    df = pd.read_parquet(ficheiro)

    df = df[df["Overflow"].isin([0, 1, 2, 3])]

    for ovf_val, df_detetor in df.groupby("Overflow"):

        for cluster_id, df_cluster in df_detetor.groupby("Cluster", dropna=True):
            
            x = df_cluster["X"].to_numpy()
            y = df_cluster["Y"].to_numpy()
            
            y_valido = 5 < x.min() and y.max() < 250  
            
            x_min, x_max = x.min(), x.max()
            x_valido = ((5 < x_min and x_max < 250) or (251 < x_min and x_max < 762) or (763 < x_min and x_max < 1018))

            if not (y_valido and x_valido):
                continue

            tot = df_cluster["ToT (keV)"].to_numpy()
            ti_ns = df_cluster["Ns"].min()
            
            n_pixeis_ativados = len(set(zip(x, y)))
            
            if n_pixeis_ativados > 30:
                fit = linregress(x, y)
                m = fit.slope
                b = fit.intercept
                r_sq = fit.rvalue ** 2

                dist_projeção_px = np.sqrt((x.max() - x.min())*2 + (y.max() - y.min())*2)
                dist_projeção_mm = dist_projeção_px * 0.055
                espessura_sensor_mm = 2.0
                
                d_mm = np.sqrt(dist_projeção_mm*2 + espessura_sensor_mm*2)
                energia_cluster = tot.sum()
                stopping_power = energia_cluster / d_mm

                cluster_info = {"ficheiro": ficheiro.name, "overflow_id": ovf_val, "cluster_id": cluster_id}
                muoes_cand.append(cluster_info)

                if r_sq >= 0.75 and (580 <= stopping_power <= 800):

                    max_x = int(x.max()) + 1
                    max_y = int(y.max()) + 1

                    heatmap = np.zeros((max_y, max_x))
                    np.add.at(heatmap, (y.astype(int), x.astype(int)), tot)
                    
                    x_esq, x_dir = x.min(), x.max()
                    y_baixo, y_cima = y.min(), y.max()
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(heatmap, origin="lower", cmap="magma_r")
                    
                    ax.set_xlim(x_esq - 1, x_dir + 1)
                    ax.set_ylim(y_baixo - 1, y_cima + 1)
                    
                    ax.set_title(f"{ficheiro.stem} | Detetor {ovf_val} - Cluster {cluster_id}\n"
                                 f"($R^2$ = {r_sq:.2f}, $d$ = {d_mm:.2f} mm, $E_{{tot}}$ = {energia_cluster:.1f} keV, $SP$ = {stopping_power:.0f} keV/mm)")
                    
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ToT (keV)")
                    plt.show()

                    resposta = input(f"\n[Ficheiro: {ficheiro.name} | Detetor: {ovf_val} | Cluster: {cluster_id}] "
                                     f"É um muão? (s/n, Enter=sim): ").strip().lower()
                    
                    if resposta in ['s', 'sim', 'y', 'yes', '']:
                        df_confirmado = df_cluster.copy()
                        
                        df_confirmado["ficheiro_origem"] = ficheiro.name
                        df_confirmado["muon_slope"] = m
                        df_confirmado["muon_intercept"] = b
                        df_confirmado["muon_r_squared"] = r_sq
                        df_confirmado["muon_distancia_d_mm"] = d_mm
                        df_confirmado["muon_energia_total_keV"] = energia_cluster
                        df_confirmado["muon_stopping_power_keV_mm"] = stopping_power
                        
                        dfs_muoes_confirmados.append(df_confirmado)
                        print("--> Muão CONFIRMADO e todas as linhas do cluster foram guardadas.")
                    else:
                        print("--> Cluster descartado pelo utilizador.")
                    
                    plt.close(fig)

print(f"\nTotal de muões confirmados manualmente: {len(dfs_muoes_confirmados)}")

if dfs_muoes_confirmados:
    df_final_muoes = pd.concat(dfs_muoes_confirmados, ignore_index=True)
    ficheiro_parquet = Path("/Users/Utilizador/Desktop/teste/clusters_muoes_completos.parquet")
    df_final_muoes.to_parquet(ficheiro_parquet, index=False)
    print(f"Ficheiro Parquet guardado em: {ficheiro_parquet}")
else:
    print("Nenhum muão foi confirmado. Nenhum ficheiro foi criado.")