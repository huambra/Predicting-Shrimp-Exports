import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

###############################################################################
# 1. MÉTRICAS OBTENIDAS DE LOS MODELOS
###############################################################################

# Resultados SARIMA (ya proporcionados en tu código original)
# Resultados Random Forest (de la imagen)
rforest_china_base = [14354.22, 15905.14, 24.31]
rforest_china_ft = [14565.94, 16003.35, 24.70]
rforest_usa_base = [4207.72, 6427.25, 19.77]
rforest_usa_ft = [3958.90, 6295.97, 18.23]
rforest_ue_base = [3171.52, 4732.29, 18.24]
rforest_ue_ft = [2979.67, 4630.26, 17.42]

# Valores ficticios para LSTM (modelo base y fine-tuned)
lstm_china_base = [0.148121, 0.172700, 17.445838]  
lstm_china_ft = [0.146891, 0.170040, 17.248605]    
lstm_usa_base = [0.118847, 0.161902, 21.968052]   
lstm_usa_ft = [0.103055, 0.151264, 18.663676]     
lstm_ue_base = [0.110872, 0.154116, 20.030747]     
lstm_ue_ft = [0.112375, 0.152459, 23.235500]   

###############################################################################
# 2. CREAR TABLA COMPARATIVA EN PANDAS
###############################################################################

model_names = [
    "SARIMA Base", "SARIMA CV", "SARIMA CV Mes a Mes", "SARIMA Log",
    "SARIMA Box-Cox", "SARIMA 2020+",
    "RF Base", "RF Fine-Tuned",
    "LSTM Base", "LSTM Fine-Tuned"
]

# Consolidar los resultados para cada país
china_metrics = [
    [6380.30, 7763.23, 12.38], [7545.44, 7545.44, 14.22], [7292.14, 8301.72, 13.85],
    [8181.65, 9858.39, 15.74], [7903.00, 9053.29, 14.94], [8838.79, 10451.99, 17.01],
    rforest_china_base, rforest_china_ft, lstm_china_base, lstm_china_ft
]

usa_metrics = [
    [2655.09, 3378.09, 13.15], [3200.04, 3200.04, 17.13], [3200.04, 3594.01, 17.13],
    [3386.28, 3852.05, 18.42], [3201.13, 3754.82, 17.05], [3289.87, 3823.54, 18.25],
    rforest_usa_base, rforest_usa_ft, lstm_usa_base, lstm_usa_ft
]

ue_metrics = [
    [1850.40, 2349.37, 11.36], [1655.75, 1655.75, 10.51], [1676.57, 1959.45, 10.47],
    [1582.40, 1854.48, 9.81], [1797.66, 2076.38, 11.35], [2028.48, 2355.94, 13.46],
    rforest_ue_base, rforest_ue_ft, lstm_ue_base, lstm_ue_ft
]

# Crear DataFrame
comparison_df = pd.DataFrame({
    "Modelo": model_names,
    "China_MAE": [x[0] for x in china_metrics],
    "China_RMSE": [x[1] for x in china_metrics],
    "China_MAPE": [x[2] for x in china_metrics],
    "USA_MAE": [x[0] for x in usa_metrics],
    "USA_RMSE": [x[1] for x in usa_metrics],
    "USA_MAPE": [x[2] for x in usa_metrics],
    "UE_MAE": [x[0] for x in ue_metrics],
    "UE_RMSE": [x[1] for x in ue_metrics],
    "UE_MAPE": [x[2] for x in ue_metrics]
})

# Ordenar columnas si lo deseas
comparison_df = comparison_df[[
    "Modelo",
    "China_MAE", "China_RMSE", "China_MAPE",
    "USA_MAE", "USA_RMSE", "USA_MAPE",
    "UE_MAE", "UE_RMSE", "UE_MAPE"
]]

###############################################################################
# 3. PUBLICAR GRÁFICOS EN STREAMLIT
###############################################################################

st.title("Comparación de Modelos")

st.write("### Tabla Comparativa de Métricas")
st.dataframe(comparison_df)

st.write("### Gráficos de MAPE")
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

ax[0].bar(model_names, comparison_df["China_MAPE"], color='skyblue')
ax[0].set_title('MAPE - China')
ax[0].set_xticklabels(model_names, rotation=45, ha='right')
ax[0].set_ylabel('MAPE (%)')

ax[1].bar(model_names, comparison_df["USA_MAPE"], color='orange')
ax[1].set_title('MAPE - USA')
ax[1].set_xticklabels(model_names, rotation=45, ha='right')

ax[2].bar(model_names, comparison_df["UE_MAPE"], color='green')
ax[2].set_title('MAPE - UE')
ax[2].set_xticklabels(model_names, rotation=45, ha='right')

st.pyplot(fig)

st.write("### Gráficos de MAE")
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

ax[0].bar(model_names, comparison_df["China_MAE"], color='skyblue')
ax[0].set_title('MAE - China')
ax[0].set_xticklabels(model_names, rotation=45, ha='right')
ax[0].set_ylabel('MAE')

ax[1].bar(model_names, comparison_df["USA_MAE"], color='orange')
ax[1].set_title('MAE - USA')
ax[1].set_xticklabels(model_names, rotation=45, ha='right')

ax[2].bar(model_names, comparison_df["UE_MAE"], color='green')
ax[2].set_title('MAE - UE')
ax[2].set_xticklabels(model_names, rotation=45, ha='right')

st.pyplot(fig)

st.write("### Gráficos de RMSE")
fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

ax[0].bar(model_names, comparison_df["China_RMSE"], color='skyblue')
ax[0].set_title('RMSE - China')
ax[0].set_xticklabels(model_names, rotation=45, ha='right')
ax[0].set_ylabel('RMSE')

ax[1].bar(model_names, comparison_df["USA_RMSE"], color='orange')
ax[1].set_title('RMSE - USA')
ax[1].set_xticklabels(model_names, rotation=45, ha='right')

ax[2].bar(model_names, comparison_df["UE_RMSE"], color='green')
ax[2].set_title('RMSE - UE')
ax[2].set_xticklabels(model_names, rotation=45, ha='right')

st.pyplot(fig)
