import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
import keras
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from io import BytesIO

your_path = r"C:\Users\Breast_Ultrasound"

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
	model = tf.keras.models.load_model(your_path + r'\best_model_2.h5')
	return model


def predict_class(image, model):
# 	image = tf.cast(image, tf.float32)
	image = np.resize(image, (224,224))
# 	image_1 = image
	image = np.dstack((image,image,image))
# 	image_2 = image
# 	cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	image = np.expand_dims(image, axis = 0)
# 	image_3 = image   


	prediction = model.predict(image)

	return prediction

def preprocessing_uploader(file, model):
    bytes_data = file.read()
    inputShape = (224, 224)
    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    prediction = model.predict(image) 
    return prediction
app_mode = st.sidebar.selectbox('Chọn trang',['Thông tin chung','Thống kê về dữ liệu huấn luyện','Ứng dụng chẩn đoán']) #two pages
if app_mode=='Thông tin chung':
    st.title('Giới thiệu về thành viên')
    st.markdown("""
    <style>
    .big-font {
    font-size:35px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .name {
    font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font"> Học sinh thực hiện </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> I. Lê Vũ Anh Tin - 9A2 </p>', unsafe_allow_html=True)
    tin_ava = Image.open(your_path + r'\member\Tin.jpg')
    st.image(tin_ava)
    st.markdown('<p class="name"> II. Trần Thị Tuyết Anh - 12A3 </p>', unsafe_allow_html=True)
    anh_ava = Image.open(your_path + r'\member\Anh.png')
    st.image(anh_ava)

    
    st.markdown('<p class="big-font"> Giáo viên hướng dẫn đề tài </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Lê Thúy Phương Như - Giáo viên Sinh Học </p>', unsafe_allow_html=True)
    Nhu_ava = Image.open(your_path + r'\member\Nhu.jpeg')
    st.image(Nhu_ava)

    st.markdown('<p class="big-font"> Trường học tham gia cuộc thi KHKT-Khởi nghiệp </p>', unsafe_allow_html=True)
    st.markdown('<p class="name"> Trường Tiểu học, THCS & THPT Hoàng Việt </p>', unsafe_allow_html=True)
    school_ava = Image.open(your_path + r'\member\school.jpg')
    st.image(school_ava)
   

elif app_mode=='Thống kê về dữ liệu huấn luyện': 
    st.title('Thống kê tổng quan về tập dữ liệu')
    
    st.markdown("""
    <style>
    .big-font {
    font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font"> I. Thông tin về tập dữ liệu </p>', unsafe_allow_html=True)
    st.caption('Tập dữ liệu ảnh siêu âm vú được lấy từ kho lưu trữ công cộng do bệnh viện Baheya, Cairo, Ai Cập cung cấp. Dữ liệu được thu thập lúc ban đầu bao gồm hình ảnh siêu âm vú ở phụ nữ trong độ tuổi từ 25 đến 75 tuổi. Số liệu này được thu thập vào năm 2018. Số lượng bệnh nhân là 600 bệnh nhân nữ. Bộ dữ liệu bao gồm 780 hình ảnh với kích thước hình ảnh trung bình là 500 * 500 pixel. Các hình ảnh có định dạng PNG. Hình ảnh được chia làm 3 loại là bình thường, lành tính và ác tính. ')
    st.caption('Nội dung nghiên cứu khoa học và ứng dụng của nhóm được thiết kế dựa trên việc huấn luyện nhóm dữ liệu Breast Ultrasound Images Dataset. Dữ liệu đã được tiền xử lý và thay đổi kích thước về 256 x 256. Thông tin chi tiết của tập dữ liệu có thể tìm được ở dưới đây: ')
    st.caption('*"https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data"*')
    covid_dataset = Image.open(your_path + r'\stat_image\Breast_Ultrasound_Images_dataset.png')
    st.image(covid_dataset)
    #Vẽ sample ảnh
    st.text('1) Một vài mẫu của ảnh siêu âm vú có khối u lành tính.')
    covid_sample = Image.open(your_path + r'\stat_image\benign_sample.png')
    st.image(covid_sample)
    
    st.text('2) Một vài mẫu của ảnh siêu âm vú có khối u ác tính.')
    non_covid_sample = Image.open(your_path + r'\stat_image\malignant_sample.png')
    st.image(non_covid_sample)
    
    st.text('3) Một vài mẫu của ảnh siêu âm vú bình thường.')
    normal_sample = Image.open(your_path + r'\stat_image\normal_sample.png')
    st.image(normal_sample)
    
    
    #Vẽ thống kê tập dữ liệu
    st.markdown('<p class="big-font"> II. Thống kê về tập dữ liệu </p>', unsafe_allow_html=True)
   
    
    data_infor = Image.open(your_path + r'\stat_image\data_infor.png')
    st.image(data_infor)
    
elif app_mode=='Ứng dụng chẩn đoán':
    model = load_model()
    st.title('Ứng dụng chẩn đoán bệnh ung thư vú dựa trên ảnh siêu âm vú')

    file = st.file_uploader("Bạn vui lòng nhập ảnh siêu âm vú để phân loại ở đây", type=["jpg", "png"])
# 

    if file is None:
        st.text('Đang chờ tải lên....')

    else:
        slot = st.empty()
        slot.text('Hệ thống đang thực thi chẩn đoán....')
        
        pred = preprocessing_uploader(file, model)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)
        class_names = ['benign', 'malignant','normal']

        result = class_names[np.argmax(pred)]
        
        if str(result) == 'benign':
            statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân có khối u lành tính.**')
            st.error(statement)
        elif str(result) == 'malignant':
            statement = str('Chẩn đoán của mô hình học máy: **Bệnh nhân mắc ung thư vú.**')
            st.warning(statement)
        elif str(result) == 'normal':
            statement = str('Chẩn đoán của mô hình học máy: **Không có dấu hiệu khối u ở vú.**')
        slot.success('Hoàn tất!')

#         st.success(output)
     
        #Plot bar chart
        bar_frame = pd.DataFrame({'Xác suất dự đoán': [pred[0,0] *100, pred[0,1]*100, pred[0,2]*100], 
                                   'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"]
                                 })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(y = 'Xác suất dự đoán', x = 'Loại chẩn đoán' )
        st.altair_chart(bar_chart, use_container_width = True)
        #Note
        st.write('- **Xác suất bệnh nhân có khối u lành tính là**: *{}%*'.format(round(pred[0,0] *100,2)))
        st.write('- **Xác suất bệnh nhân mắc ung thư vú là**: *{}%*'.format(round(pred[0,1] *100,2)))
        st.write('- **Xác suất bệnh nhân khỏe mạnh là**: *{}%*'.format(round(pred[0,2] *100,2)))
 

        
        
        
        
        
        
        
        
        
        
        
        
        