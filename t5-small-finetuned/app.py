"""Streamlit interface for the recovered T5-small summarizer."""
import os
import logging
import streamlit as st
from model_runtime import load_runtime, summarize, MAX_INPUT_CHARACTERS

st.set_page_config(page_title='T5 News Summarizer', page_icon='📰', layout='centered')
st.title('T5 News Summarizer')
st.write('Turn an English news article into a short summary using a recovered T5-small checkpoint.')
st.caption('An individual university fine-tuning project by Faizan Tariq. Summaries can omit context or make mistakes.')


def setting(name):
    if name in os.environ:
        return os.environ[name]
    # Avoid rendering a secrets-file error when local inference needs no secrets.
    if st.secrets.load_if_toml_exists():
        return str(st.secrets.get(name, ''))
    return ''


@st.cache_resource(show_spinner=False)
def cached_runtime(local_dir, repo_id, revision):
    return load_runtime(local_dir, repo_id, revision)


SAMPLE = ('The city library will extend its opening hours starting next Monday. The building will stay open '
          'until 9 p.m. on weekdays, two hours later than before. Library director Sara Khan said the change '
          'follows requests from students and people who work during the day. Weekend hours will remain '
          'unchanged. The council approved funding for two additional evening staff members. The extended '
          'schedule will run for six months before officials review visitor numbers and decide whether to continue it.')


def use_example():
    st.session_state['article'] = SAMPLE
    st.session_state.pop('result', None)


st.button('Use example article', on_click=use_example)
with st.form('summarization'):
    text = st.text_area('Article', key='article', height=240, max_chars=MAX_INPUT_CHARACTERS,
                        placeholder='Paste an English news article here…')
    st.caption('The model reads up to 512 tokens, including its instruction. Longer articles are truncated.')
    with st.expander('Generation settings'):
        max_length = st.slider('Maximum output tokens', 32, 128, 128, step=16)
        num_beams = st.slider('Beam search width', 1, 4, 4,
                              help='Higher values explore more candidate summaries and can take longer.')
    submitted = st.form_submit_button('Generate summary', type='primary')

if submitted:
    st.session_state.pop('result', None)
    if not text.strip():
        st.warning('Paste an article before generating a summary.')
    else:
        try:
            with st.spinner('Loading the model and generating your summary. The first request takes longer…'):
                runtime = cached_runtime(setting('T5_MODEL_DIR'), setting('T5_MODEL_REPO'), setting('T5_MODEL_REVISION'))
                st.session_state['result'] = summarize(runtime, text, max_length, num_beams)
        except Exception as exc:
            logging.error("T5 request failed (%s)", type(exc).__name__)
            st.error('The model could not complete this request. If you maintain this app, check the model '
                     'configuration, artifact checksums and available memory described in the deployment guide. '
                     'Otherwise, try again later.')

if 'result' in st.session_state:
    result = st.session_state['result']
    if result['truncated']:
        st.warning('This article exceeded the model input limit. Only its first 512 tokens were used.')
    st.subheader('Summary')
    st.write(result['summary'])
    st.download_button('Download summary', result['summary'], 'summary.txt', mime='text/plain')

with st.expander('About this project'):
    st.write('T5-small uses an encoder–decoder Transformer to generate text from text. This app adds the '
             'summarize: instruction and runs the recovered checkpoint on CPU. It uses no paid inference API.')
    st.write('The original project report describes CNN/DailyMail fine-tuning. Its ROUGE scores have not '
             'been independently reproduced, and the training notebook is unavailable.')
    st.caption('Please verify facts against the original article. Submitted text is processed by this app; '
               'it is not sent to a separate text-generation API.')
