import streamlit as st
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights

st.write("""
# Нейросеть Inception v.3

Нейросеть [сверточного типа](https://ru.wikipedia.org/wiki/%D0%A1%D0%B2%D1%91%D1%80%D1%82%D0%BE%D1%87%D0%BD%D0%B0%D1%8F_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C) 
от компании Google, представленная в 2015 году. Подробное 
описание можно найти в статье авторов [здесь](https://arxiv.org/pdf/1512.00567.pdf), а также на 
[хабре](https://habr.com/ru/post/302242/).

Нейросеть классифицирует изображенное на фотографии кожное образование как доброкачественное или злокачественное.
""")

model = torch.load('model_skin_cancer',map_location=torch.device('cpu'))
model.eval()
st.write('''

''')

trans = T.Compose([T.Resize((299,299)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

if "np_img" in st.session_state:
    
    img_arr = st.session_state["np_img"]
  
    # image transformation
    img_tens = torch.Tensor(img_arr) 
    img_tens = torch.permute(img_tens, (2, 0, 1))
    

    ready_img = trans(img_tens.float()/255).unsqueeze(0)
    result = round(torch.sigmoid(model(ready_img)).item())
    labels = {0: 'доброкачественное', 1: 'злокачественное'}
    
    prob = torch.sigmoid(model(ready_img)).item()
    probability = prob if result == 1 else 1 - prob
    probability = round(probability*100, 1)
    if result == 1: 
        st.markdown(f''' Результат нейросети: **:red[{labels[result]}]**''')
    else:
        st.write(f'Результат нейросети: **:blue[{labels[result]}]**' )    
    st.write('`Вероятность:`', probability)
    
    # image display
    fig, ax = plt.subplots(1,1)
    ax.imshow(img_arr)
    ax.axis('off')
    st.pyplot(fig)

    with st.expander('Техническая информация о модели'):
        st.caption('Модель дообучена двадцатью циклами обучения до точности ~81%')
        st.image('inception_metrics.png')

else:
    st.caption('Для получения результата от нейросети загрузите изображение на главной странице.')

        