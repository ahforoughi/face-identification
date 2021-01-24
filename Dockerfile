FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY face_recognition.py .

CMD [ "python","-u","./face_recognition.py" ]