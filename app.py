else:
    st.markdown("---")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        ### 📊 Sobre o Fica de Olho
        
        O **Fica de Olho** é uma ferramenta de análise financeira que extrai, processa e visualiza dados 
        de instituições financeiras brasileiras de forma automatizada e interativa.
        
        #### 🎯 Funcionalidades
        
        - **Extração Automatizada**: Integração direta com a API IF.data do Banco Central do Brasil
        - **Análise Temporal**: Acompanhamento de métricas financeiras ao longo de múltiplos trimestres
        - **Visualização Interativa**: Gráficos de dispersão customizáveis com filtros dinâmicos
        - **Classificação Personalizada**: Sistema de aliases para renomear e categorizar instituições
        - **Métricas Calculadas**: ROE anualizado, alavancagem, funding gap, market share e índices de risco/retorno
        
        #### 📈 Dados Utilizados
        
        Todos os dados são extraídos da **API IF.data** do Banco Central do Brasil, incluindo:
        
        - Carteira de Crédito Classificada
        - Patrimônio Líquido e Lucro Líquido
        - Índice de Basileia
        - Captações e Ativo Total
        - Cadastro de Instituições Financeiras
        
        #### 🚀 Como Começar
        
        1. Configure o período de análise na barra lateral
        2. Clique em **"Extrair Novos Dados"** para buscar informações do BCB
        3. Acesse os dashboards no menu lateral após a extração
        4. Personalize visualizações e exporte análises
        """)
    
    with col2:
        st.info("""
        ### 💡 Primeira Vez?
        
        **Passo 1:** Configure as datas na barra lateral ←
        
        **Passo 2:** Clique em "🚀 Extrair Novos Dados"
        
        **Passo 3:** Aguarde o processamento (30-60 segundos)
        
        **Passo 4:** Explore os dashboards!
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 Recursos Técnicos
        
        - **Python 3.10+**
        - **Streamlit** (interface)
        - **Pandas** (processamento)
        - **Plotly** (visualizações)
        - **API BCB Olinda**
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666; font-size: 14px;'>
        Desenvolvido em 2026 por <strong>Matheus Prates, CFA</strong><br>
        Ferramenta de código aberto para análise do sistema financeiro brasileiro
    </div>
    """, unsafe_allow_html=True)
