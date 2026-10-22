import streamlit as st
import inference
import plotly.graph_objects as go
import catboost

@st.cache_resource(show_spinner=False)
def get_coffee_prediction(coffee_data):
    rag_model, rag_tokenizer = inference.init_model_for_rag()
    catboost_model = catboost.CatBoostClassifier()
    catboost_model.load_model("models/catboost_model.cbm")
    return inference.predict_coffee_taste(coffee_data, catboost_model, rag_model, rag_tokenizer)

st.write("Get your coffee taste")

countries_list = ['Ethiopia', 'Guatemala', 'Brazil', 'Peru', 'Hawaii',
    'Indonesia', 'China', 'Costa Rica', 'Mexico', 'Uganda',
    'Honduras', 'Taiwan', 'Nicaragua', 'Tanzania',
    'Kenya', 'Thailand', 'Colombia', 'Panama', 'Papua New Guinea', 'El Salvador', 'Japan',
    'Ecuador', 'Puerto Rico','Haiti', 'Burundi', 'Vietnam', 'Philippines', 'Rwanda', 'Malawi',
    'Laos', 'Zambia', 'Myanmar', 'Mauritius', 'Malaysia', 'India', 'Bolivia','Jamaica', 'Congo', 'Yemen',
    'Zambia', 'Nepal', 'Zimbabwe', 'South and Central America', 'Other']

processing_method = ['washed', 'natural', 'honey', 'anaerobic', 'carbonic', 'co-fermented',
                     'wet-hulled', 'other']

variety_list = [
    'Bourbon', 'Catimor', 'Caturra',
    'SL14', 'Sumatra', 'SL34', 'Hawaiian Kona', 'Yellow Bourbon',
    'SL28', 'Geisha', 'Catuai', 'Catucai', 'Pacamara', 'Typica', 'Sumatra Lintong',
    'Mundo Novo', 'Java', 'Peaberry', 'Pacas', 'Mandheling',
    'Ruiru 11', 'Arusha', 'Ethiopian Heirlooms', 'Moka' , 'Peaberry',
    'Sulawesi', 'Blue Mountain', 'Marigojipe', 'Pache Comun', 'Guji',
    'SL09', 'Yirgacheffe', 'Sidamo', 'Sidra', 'Huila', 'Castillo', 'Anacafe 14', 'Arabica'
]

roast_type = ['Medium-Light', 'Medium', 'Medium-Dark', 'Light', 'Dark', 'Very Dark', 'Other']

all_categories = ['roasted', 'spices', 'nutty/cocoa', 'sweet', 'floral', 'fruit', 'sour/fermented', 'green/vegetative', 'other']
bright_colors = [
    "#d33727", "#b90d41", "#9a7b79", "#f36421", "#ec008c",
    "#ee1d23", "#f5ce02", "#17803b", "#00a7d2"
]
grey_color = "#E0E0E0"

country_of_origin = st.selectbox("Choose country of origin", countries_list)
roast_type = st.selectbox("Choose roast type", roast_type)
processing_method = st.selectbox("Choose processing method", processing_method)
variety_list = st.multiselect("Choose coffee variety", variety_list)

if st.button("Ok", type="primary"):
    coffee_data = {
        "country_of_origin": country_of_origin,
        "processing_method": processing_method,
        "variety": variety_list,
        "roast": roast_type
    }

    with st.spinner('Getting the result...'):
        try:
            taste_notes, description = get_coffee_prediction(coffee_data)
            chart_labels = []
            chart_colors = []
            for i, cat in enumerate(all_categories):
                chart_labels.append(cat)
                if cat in taste_notes[0]:
                    chart_colors.append(bright_colors[i])
                else:
                    chart_colors.append(grey_color)
            fig = go.Figure(data=[go.Pie(
                labels=chart_labels,
                hole=0.3,
                sort=False
            )])

            fig.update_traces(
                marker=dict(colors=chart_colors, line=dict(color='#FFFFFF', width=2)),
                textinfo='label',
                hoverinfo='label'
            )

            fig.update_layout(
                title_text="Results",
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            st.text_area("Result", value=description, height=150)
        except Exception as e:
            st.error(f"Error: {e}")
