FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install pipenv
ENV PROJECT_DIR ./
WORKDIR ${PROJECT_DIR}

COPY predict.py ${PROJECT_DIR}/
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

# RUN python predict.py
# ENTRYPOINT ["python", "predict.py"]