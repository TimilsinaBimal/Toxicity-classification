from fastapi import FastAPI, Body
from prediction import predict

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Server is up and running!"}


@app.post("/predict/")
def predict_label(text: str = Body()):
    return predict(text)
