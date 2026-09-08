INSTRUCTIONS

The file 'muoes.py' uses the mentioned criteria and displays muon candidates. The user observes the activated pixels, as well as other characteristics, and decides whether it is indeed a muon or not. Use 's' (or Enter) for "yes" and 'n' for "no". Don't forget to change the path to the desired directory. The muons and all their characteristics are stored in Parquet format under the name "clusters_muoes_completos_source.parquet".

The "tempos_relativos_source.py" files calculate the relative times of each activated pixel for each of the radioactive sources. The data is stored in "lista_muoes_completa_source.parquet".

"gráficos_source.py" performs a linear fit of the Z(t_relative) plot.

"grafico_vderiva_z_por_chip_background_wPb_2" uses the display window and removes data points that deviate beyond a desired multiple of the standard deviation.

