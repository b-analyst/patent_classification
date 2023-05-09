from typing import List
import streamlit as st
import requests
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch.nn as nn
from clf.frontend.sidebar import Sidebar

st.session_state['recall'] = 0
st.session_state['precision'] = 0

class App:
    def __init__(self):
        self.sidebar = Sidebar()

        
    def __call__(self):
        self.sidebar()
        self.main()

    def main(self):
        st.title(':red[TechNext.AI]')
        st.header('Patent Classification System v2')
        with st.form('patent classifier'):
            inp = st.text_area(label='Enter patent abstract')
            if st.form_submit_button():
                st.snow()
                detected = self.inference(self.sidebar.model, inp)
                st.write(detected)

        with st.expander('Validate Classifier'):
            data = json.dumps(
                {
                    'model': self.sidebar.model,
                    'stage_1_thresh': int(self.sidebar.stage_1_thresh),
                    'stage_2_thresh': float(self.sidebar.stage_2_thresh)
                }
            )
            if st.button('run'):
                res = requests.post(url='http://127.0.0.1:8888/validate/', data=data)
                res = res.json()
                recall = res[0]
                precision = res[1]
                recall_delta = recall - st.session_state['recall']
                precision_delta = precision - st.session_state['precision']
                st.metric(label='recall', value=recall, delta=recall_delta)
                st.metric(label='precision', value=precision, delta=precision_delta)
                st.session_state['recall'] = recall
                st.session_state['precision'] = precision

    def inference(self, model, inp):
        data = json.dumps(
            {
                'model': model,
                'inp': inp,
                'stage_1_thresh': int(self.sidebar.stage_1_thresh),
                'stage_2_thresh': float(self.sidebar.stage_2_thresh)
            }
        )
        res = requests.post(url='http://127.0.0.1:8888/inference/', data=data)
        return res.json()
    
    
if __name__ == '__main__':
    app = App()
    app()