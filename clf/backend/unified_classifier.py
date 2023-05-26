import os
from typing import List, Union, Optional
import numpy as np
from tqdm.auto import tqdm
import pickle
import pandas as pd
from backend.scoring import Scoring
from sentence_transformers import SentenceTransformer, util
import keras
from loguru import logger
import json
import random
import ast
import gc
# import torch
# from keras import backend as K
import tensorflow.compat.v1 as tf
# from numba import cuda


data_source = pd.read_csv(os.path.join(os.getcwd(), 'backend/data/patents_v7.csv'))
errors = []
validation = pd.read_csv(os.path.join(os.getcwd(), 'backend/data/grouped_labels.csv')).sample(n=20)
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


class UnifiedClassifier:
    def __init__(self, model: Optional[str]=None):
        '''
        Initialize with path to directory containing folders for each class.
        In each class folder, there should be two .pk files called:
            {class}_classifier.pk
            {class}_vectorizer.pk
        
        Subject to change in the future as we move to cloud storage.

        This class is a utility for taking in a patent abstract and running it through all 425 classifiers.
        The result is a multilabel classification labeling the categories a certain patent falls into.        
        '''
        self.stage_1_path = os.path.join(os.getcwd(), 'backend/mnb_models_v2')
        self.stage_2_path = os.path.join(os.getcwd(), 'backend/subclass_models')
        
        if model is None:
            self.similarity = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        elif model:
            self.similarity = SentenceTransformer(model, device='cuda')
        self.similarity.max_seq_length = 512
        
    def _load_vectorizer(self, clss: int):
        cls_dir = os.path.join(self.stage_1_path, clss)
        try:
            vectorizer = pickle.load(open(os.path.join(cls_dir, f'{clss}_vectorizer.pk'), 'rb'))
            return {
                'vectorizer': vectorizer
            }
        except Exception:
            logger.exception('The specified vectorizer could not be found at this location.')
        
    def _load_classifier(self, clss: int):
        cls_dir = os.path.join(self.stage_1_path, clss)
        try:
            classifier = pickle.load(open(os.path.join(cls_dir, f'{clss}_classifier.pk'), 'rb'))
            return {
                'classifier': classifier
            }
        except Exception:
            logger.exception('The specified classifier could not be found at this location.')
        
    def _featurize(self, data: str, clss: int):
        load = self._load_vectorizer(clss)
        # logger.info(f'loaded {clss} vectorizer. Vectorizing...')
        return load['vectorizer'].transform([data])
    
    def _scoring(self, inp: str, detected: List[int]):
        # for classes in detected:
        cls_map = {
            cls: data_source[data_source['mainclass_id'] == cls].sample(n=30)['patent_text'].to_list()
            for cls in detected
        }
        similarity_scores = {}
        inp_encoding = self.similarity.encode(inp)

        for k, v in tqdm(cls_map.items()):
            encoded = [self.similarity.encode([np.array(v)])]
            scores = [util.cos_sim(inp_encoding, enc)[:].item() for enc in encoded]
    #         print(scores)
            score = sum(scores)/len(scores)
            
            similarity_scores[k] = score

        return similarity_scores
    
    def _dropout(self, clss: List[int], dropout: float=0.35):
        random.shuffle(clss)
        drop = round(len(clss) * dropout)
        return clss[:-drop]
    
    def stage_1_predict(self, data: str, stage_1_thresh: int=25):
        classes = []
        for clss in tqdm(os.listdir(self.stage_1_path)):
            features = self._featurize(data, clss)
            load = self._load_classifier(clss)
            res = load['classifier'].predict(features)
            proba = load['classifier'].predict_proba(features)
            # print(proba)
            if res != 0 and proba[0][1] > .9:
                # proba = pd.DataFrame(load['classifier'].predict_proba(features), columns=load['classifier'].classes_)
                # classes.append(proba)
            # if res != 0:
                classes.append(res[0])
        # clss, sims = [],[]
        clss = []
        similarity_scores = self._scoring(data, classes)
        temp = pd.DataFrame()
        temp['classes'] = list(similarity_scores.keys())
        temp['similarity'] = list(similarity_scores.values())
        temp = temp.sort_values(by=['similarity'])
        clss.extend(temp['classes'].to_list()[-stage_1_thresh:])
        # clss = self._dropout(clss)
        # sims.append(temp['similarity'].to_list()[-15:])
        gc.collect()
        return clss
    
    def stage_2_predict(self, data: str, clss: List[Union[str, int]], stage_2_thresh: float=.05):
        predict = []
        log = None
        
        
        for cls in clss:
            if os.path.exists(os.path.join(self.stage_2_path, f'{str(cls)}/{str(cls)}_mlb.pkl')):
                final = []
                with open(os.path.join(self.stage_2_path, f'{str(cls)}/{str(cls)}_mlb.pkl'), 'rb') as f:
                    mlb = pickle.load(f) 
                logger.info(f'loaded {cls} mlb')
                embeddings = self.similarity.encode([data])

                print('setting new session')

                model = keras.models.load_model(os.path.join(self.stage_2_path, f'{str(cls)}/{str(cls)}.ckpt'))
                logger.info(f'loaded {cls} classifier')
                # preds = model.predict(embeddings)
                print('using __call__ method')
                preds = model(tf.convert_to_tensor(embeddings, dtype=tf.float32))
                    
                final.append(preds)
                Test_prob = np.mean(final, 0)
                try:
                    print(len(mlb.classes_))
                    Test_prob = pd.DataFrame(Test_prob, columns=list(mlb.classes_))
                    predict.append(list(Test_prob[Test_prob >= stage_2_thresh].dropna(axis=1).columns))
                except:
                    log = {
                        "class": cls,
                        "output_dim": len(Test_prob[0]),
                        "mlb_dim": len(mlb.classes_)
                    }
                    if log not in errors:
                        errors.append(log)
                    
            else:
                predict.append(cls)
                continue
            # predict.append(list(Test_prob[Test_prob >= .15].dropna(axis=1).columns))

            if log:
                with open('errors.json', 'w') as f:
                    log = json.dumps(errors, indent=4)
                    f.write(log)
        # predict = self._dropout(predict)
        # K.clear_session()
        # torch.cuda.empty_cache()
        # gc.collect()
        # session.close()
        return predict
    
    # def clear_session(self):
    #     K.clear_session()
    #     torch.cuda.empty_cache()
    #     # dev = cuda.get_current_device()
    #     # dev.reset()
    #     # cuda.close()
    #     gc.collect()
    
    def clean_result(self, predictions):
        results = []  
        for pred in predictions:
            try:
                int(pred)
                results.append(pred)
            except:
                for p in pred:
                    results.append(p)
        return results
    
    def stage_1(self, data, stage_1_thresh):
        return self.stage_1_predict(data, stage_1_thresh)
    
    def predict(self, data: str, stage_1_thresh: int, stage_2_thresh: float):
        clss = self.stage_1_predict(data, stage_1_thresh)
        predict = self.stage_2_predict(data, clss, stage_2_thresh)
        return self.clean_result(predict)

    def validate(self, stage_1_thresh: int, stage_2_thresh: float):
        recall, precision = [], []
        for _, row in validation.iterrows():
            res = self.predict(row['patent_text'], stage_1_thresh, stage_2_thresh)
            scoring = Scoring(res, ast.literal_eval(row['subclass_id']))
            recall.append(scoring.recall())
            precision.append(scoring.precision())
        
        avg_recall = sum(recall)/len(recall)
        avg_precision = sum(precision)/len(precision)

        return avg_recall, avg_precision
            

