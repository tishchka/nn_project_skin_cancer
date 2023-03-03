import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


st.write("""
# Визуальный анализ кожных новообразований с помощью нейросетей 

Злокачественные новообразования кожи – распространенная патология во всем мире.


В России рак кожи по частоте встречаемости, по данным за 2020 год, занимает у женщин второе место – 14,5 %, 
у мужчин третье место – 10,6 % [(источник)](https://www.niioncologii.ru/highlights/index?id=9643)

Визуальная диагностика - первый и самый простой способ заподозрить наличие новообразования. 
Врач производит осмотр и в случае подозрений назначает дополнительные исследования.

Дополнительным инструментом первичного анализа могут выступать нейросети - компьютерные модели,
обученные распознавать изменения в кожном покрове по фотографии.

В данном веб-приложении представлено две модели анализа кожных новообразований. 

Обучение моделей проводилось на датасете размеченных фотографий (то есть с указанием статуса - доброкачественное/злокачественное образование),
с сайта [kaggle.com](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?datasetId=174469&searchQuery=pyt)


""")
st.caption('''
*Для использования моделей загрузите изображение на панели слева. Затем перейдите на страницу модели.*
''')
           
st.sidebar.header('Загрузите фотографию')

#def img_plot(img_arr):
    

uploaded_file = st.sidebar.file_uploader(
    "Можно перетащить (драг-н-дроп)", 
    type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None:
    img_arr = np.array(Image.open(uploaded_file).convert('RGB'))
    st.write('''
        ### Загруженная фотография:
        ''')
    fig, ax = plt.subplots(1,1)
    ax.imshow(img_arr)
    ax.axis('off')
    st.pyplot(fig)
    st.session_state["np_img"] = img_arr
    

elif "np_img" in st.session_state:
    img_arr = st.session_state['np_img'] 

    st.write('''
        ### Загруженная фотография:
        ''')
    fig, ax = plt.subplots(1,1)
    ax.imshow(img_arr)
    ax.axis('off')
    st.pyplot(fig)

