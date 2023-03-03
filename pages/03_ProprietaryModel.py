import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import transforms as T
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights

st.write("""
# Нейросеть Pooling

Нейросеть [сверточного типа](https://ru.wikipedia.org/wiki/%D0%A1%D0%B2%D1%91%D1%80%D1%82%D0%BE%D1%87%D0%BD%D0%B0%D1%8F_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C) 
разработанная группой студентов [ElbrusBootcamp](https://elbrusboot.camp/datascience/).

Нейросеть классифицирует изображенное на фотографии кожное образование как доброкачественное или злокачественное.
""")


class BaseCNN(nn.Module):
    def __init__(self) -> None:
        super(BaseCNN,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, # (w - k +2*p) / s + 1
                                              out_channels=7, 
                                              kernel_size=5), 
                                    
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        
        

        self.layer2 = nn.Sequential(nn.Conv2d(7, 16, 
                                              kernel_size=5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
                                    
        # self.flatten =  nn.Flatten()                     
                                    
        self.fc1 = nn.Linear(13456, 1280) 
        self.drop_out = nn.Dropout() 
        self.fc2 = nn.Linear(1280, 256)
        self.fc3 = nn.Linear(256, 1)
        self.out_m = 0

            
    def forward(self, x): 
        out = self.layer1(x)
        out_first = out
        
        # сохраняем feature maps после слоя 1 для визуализации
        # out_first = out
        
        out = self.layer2(out)
        
        
        # # сохраняем feature maps после слоя 2 для визуализации
        out_second = out
        out = out.reshape(out.size(0), -1)
        # out = self.flatten(out)
        out = self.drop_out(out)
        out = torch.relu(self.fc1(out))
        out = self.drop_out(out) 
        out = torch.relu(self.fc2(out))
        out = self.drop_out(out)
        out = self.fc3(out)
        
        return out

model = torch.load('my_model_skin')#, map_location=torch.device('cpu'))
model.eval()
st.write('''

''')

trans = T.Compose([T.Resize((128,128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
        st.caption('Модель обучена десятью циклами обучения до точности ~84%')
        st.image('pooling_metrics.png')

else:
    st.caption('Для получения результата от нейросети загрузите изображение на главной странице.')

        