import streamlit as st

def run_app():
    # Page Config
    st.set_page_config(
        page_title="Streamlit ML Lab: Accessible Machine Learning",
        page_icon = ":mag:",
        initial_sidebar_state = 'auto'
    )

    # Content
    st.sidebar.title("Streamlit ML Lab: Accessible Machine Learning")

    # Multipages and Navigation
    pred_page = st.Page("home/prediction.py", title="Prediction")

    data_pre_page = st.Page("dataset/data_preprocessing.py", title="Dataset")
    eda_ua_page = st.Page("dataset/eda_ua.py", title="EDA: Univariate Analysis")
    eda_ba_page = st.Page("dataset/eda_ba.py", title="EDA: Bivariate Analysis")

    train_page = st.Page("modeling/training.py", title="Training")
    eval_page = st.Page("modeling/evaluation.py", title="Evaluation")
    explained_page = st.Page("modeling/explainability.py", title="Explainability")

    pg = st.navigation({
        "Home": [pred_page],
        "Dataset": [data_pre_page, eda_ua_page, eda_ba_page],
        "Modeling": [train_page, eval_page, explained_page],   
    })

    pg.run()

    # Session State
    # st.write("=======================")
    # st.write("Session State")
    # st.write(st.session_state)
    # st.write("=======================")

if __name__ == "__main__":
    # App
    run_app()