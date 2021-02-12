FROM python:3
FROM tensorflow/tensorflow:latest

WORKDIR /usr/src/app



COPY requirements.txt ./ 
RUN pip install autocrop
RUN pip install --no-cache-dir -r requirements.txt



COPY face_rec.py .
COPY arc_face.py .
COPY preprocess.py . 

RUN apt install -y libgl1-mesa-glx

CMD ["python", "-u", "./face_rec.py"]   

