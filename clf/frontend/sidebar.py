import streamlit as st

class Sidebar:
    def __call__(self):
        with st.sidebar:
            st.title("Settings")
            self.model = st.selectbox(
                'Which Sentence Transformer model would you like to use?',
                ('all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'Paraphrase-MiniLM-L3-v2')
            )
            self.stage_1_thresh = st.slider(
                'select stage 1 threshold',
                1, 30, 15
            )
            self.stage_2_thresh = st.slider(
                'select stage 2 threshold',
                0.01, 0.5, 0.05
            )
            
            
