import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Scatter Plot", page_icon="📊", layout="wide")

if 'dados_periodos' not in st.session_state or not st.session_state['dados_periodos']:
    st.warning("⚠️ Extraia os dados primeiro na página principal")
    st.stop()

df_completo = pd.concat(st.session_state['dados_periodos'].values(), ignore_index=True)
df_completo = df_completo[df_completo['Instituição'].notna()].copy()
dict_cores = st.session_state.get('dict_cores_personalizadas', {})

st.title("📊 Escolha os Eixos X e Y do gráfico!")

def is_percentage_column(col_name):
    percentage_keywords = ['%', 'ROE', 'Basileia', 'Gap', 'Ativo']
    return any(keyword in col_name for keyword in percentage_keywords)

# Inicializar cores customizadas no session_state
if 'cores_customizadas' not in st.session_state:
    st.session_state['cores_customizadas'] = dict_cores.copy()

with st.sidebar:
    st.header("🎯 Configurações")
    
    periodos = sorted(df_completo['Período'].unique(), key=lambda x: (x.split('/')[1], x.split('/')[0]))
    periodo = st.selectbox("📅 Período", periodos, index=len(periodos)-1)
    
    st.divider()
    
    colunas = [c for c in df_completo.columns if df_completo[c].dtype in ['float64','int64']]
    var_x = st.selectbox("Eixo X", colunas, index=colunas.index('Alavancagem') if 'Alavancagem' in colunas else 0)
    var_y = st.selectbox("Eixo Y", colunas, index=colunas.index('Índice de Basileia') if 'Índice de Basileia' in colunas else 1)
    
    st.divider()
    
    bancos = sorted(df_completo['Instituição'].unique())
    tipo = st.radio("Filtrar", ["TOP 15", "Todos", "Selecionar"])
    
    if tipo == "Selecionar":
        selecionados = st.multiselect("Bancos", bancos, default=bancos[:10])
    elif tipo == "TOP 15":
        top = st.slider("TOP N", 5, 50, 15)
        selecionados = df_completo[df_completo['Período']==periodo].nlargest(top, 'Carteira de Crédito')['Instituição'].tolist()
    else:
        selecionados = bancos
    
    st.divider()
    
    tamanho = st.slider("Tamanho pontos", 8, 25, 12)
    
    st.divider()
    
    # EDITOR DE CORES
    st.subheader("🎨 Personalizar Cores")
    
    mostrar_editor = st.checkbox("Editar cores dos bancos", value=False)
    
    if mostrar_editor and selecionados:
        st.caption("Escolha a cor de cada banco:")
        
        cores_padrao = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, banco in enumerate(selecionados[:10]):  # Limitar a 10 para não ficar muito longo
            cor_atual = st.session_state['cores_customizadas'].get(banco, cores_padrao[i % len(cores_padrao)])
            nova_cor = st.color_picker(banco, value=cor_atual, key=f"cor_{banco}")
            st.session_state['cores_customizadas'][banco] = nova_cor
        
        if len(selecionados) > 10:
            st.caption(f"Mostrando apenas os 10 primeiros de {len(selecionados)} bancos")
        
        if st.button("🔄 Resetar Cores", use_container_width=True):
            st.session_state['cores_customizadas'] = dict_cores.copy()
            st.rerun()

df = df_completo[(df_completo['Período']==periodo) & (df_completo['Instituição'].isin(selecionados))].dropna(subset=[var_x, var_y])

if df.empty:
    st.error("❌ Sem dados")
    st.stop()

df_plot = df.copy()

x_is_pct = is_percentage_column(var_x)
y_is_pct = is_percentage_column(var_y)

if x_is_pct and df_plot[var_x].max() <= 10:
    df_plot[f'{var_x}_display'] = df_plot[var_x] * 100
    x_label = f"{var_x} (%)"
    x_format = ':.1f'
else:
    df_plot[f'{var_x}_display'] = df_plot[var_x]
    x_label = var_x
    x_format = ':.2f'

if y_is_pct and df_plot[var_y].max() <= 10:
    df_plot[f'{var_y}_display'] = df_plot[var_y] * 100
    y_label = f"{var_y} (%)"
    y_format = ':.1f'
else:
    df_plot[f'{var_y}_display'] = df_plot[var_y]
    y_label = var_y
    y_format = ':.2f'

# Usar cores customizadas
fig = px.scatter(
    df_plot, 
    x=f'{var_x}_display', 
    y=f'{var_y}_display', 
    color='Instituição', 
    hover_name='Instituição',
    hover_data={
        f'{var_x}_display': x_format,
        f'{var_y}_display': y_format,
        'Carteira de Crédito': ':,.0f',
        'Instituição': False
    },
    title=f"{y_label} vs {x_label} ({periodo})", 
    color_discrete_map=st.session_state['cores_customizadas'],
    labels={
        f'{var_x}_display': x_label,
        f'{var_y}_display': y_label
    }
)

fig.update_traces(marker=dict(size=tamanho, line=dict(width=1, color='white')))
fig.update_layout(
    height=650, 
    template='plotly_white', 
    showlegend=True, 
    hovermode='closest',
    xaxis_title=x_label,
    yaxis_title=y_label
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)

x_mean = df[var_x].mean()
y_mean = df[var_y].mean()

if x_is_pct and df[var_x].max() <= 10:
    x_mean_display = f"{x_mean * 100:.1f}%"
else:
    x_mean_display = f"{x_mean:.2f}"

if y_is_pct and df[var_y].max() <= 10:
    y_mean_display = f"{y_mean * 100:.1f}%"
else:
    y_mean_display = f"{y_mean:.2f}"

col1.metric("Bancos", len(df))
col2.metric(f"{var_x} Médio", x_mean_display)
col3.metric(f"{var_y} Médio", y_mean_display)

with st.expander("📋 Dados"):
    df_display = df[['Instituição', var_x, var_y, 'Carteira de Crédito']].copy()
    
    if x_is_pct and df[var_x].max() <= 10:
        df_display[var_x] = df_display[var_x].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    else:
        df_display[var_x] = df_display[var_x].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    if y_is_pct and df[var_y].max() <= 10:
        df_display[var_y] = df_display[var_y].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    else:
        df_display[var_y] = df_display[var_y].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    df_display['Carteira de Crédito'] = df_display['Carteira de Crédito'].apply(lambda x: f"R$ {x/1e9:.2f}B" if pd.notna(x) else "-")
    
    st.dataframe(df_display.sort_values(var_y, ascending=False), hide_index=True, use_container_width=True)
