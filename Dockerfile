FROM python:3.10
RUN apt-get update

RUN pip install contourpy==1.0.7
RUN pip install cycler==0.11.0
RUN pip install fonttools==4.38.0
RUN pip install joblib==1.2.0
RUN pip install kiwisolver==1.4.4
RUN pip install matplotlib==3.6.3
RUN pip install numpy==1.24.1
RUN pip install packaging==23.0
RUN pip install pandas==1.5.2
RUN pip install Pillow==9.4.0
RUN pip install pip==22.3.1
RUN pip install pyparsing==3.0.9
RUN pip install python-dateutil==2.8.2
RUN pip install pytz==2022.7
RUN pip install scikit-learn==1.2.0
RUN pip install scipy==1.10.0
RUN pip install setuptools==65.5.1
RUN pip install six==1.16.0
RUN pip install threadpoolctl==3.1.0
RUN pip install wheel==0.38.4
RUN pip install xgboost==1.7.3