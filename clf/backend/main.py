from fastapi import FastAPI
from backend.models import Abstract, Validation, Stage_1, Stage_2
from backend.unified_classifier import UnifiedClassifier
from typing import List, Optional
import uvicorn

app = FastAPI()

@app.post('/inference/')
async def inference(data: Abstract) -> List:
    classifier = UnifiedClassifier(data.model)
    return classifier.predict(data.inp, stage_1_thresh=data.stage_1_thresh, stage_2_thresh=data.stage_2_thresh)

@app.post('/stage_1/')
async def stage_1(data: Stage_1):
    classifier = UnifiedClassifier(data.model)
    return classifier.stage_1(data.inp, data.stage_1_thresh)

@app.post('/stage_2/')
async def stage_2(data: Stage_2):
    classifier = UnifiedClassifier()
    return classifier.stage_2_predict(data.data, data.clss, data.stage_2_thresh)

@app.post('/validate/')
async def validate(data: Validation):
    classifier = UnifiedClassifier(data.model)
    avg_recall, avg_precision = classifier.validate(data.stage_1_thresh, data.stage_2_thresh)
    return avg_recall, avg_precision






if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8888, log_level="info", reload=True)
    server = uvicorn.Server(config)
    server.run()