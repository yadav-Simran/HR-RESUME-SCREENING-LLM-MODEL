import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid
import os

#Creating session variables
if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] =''

def main():
    load_dotenv()

    st.set_page_config(page_title="Resume Screening Assistance 🕵️‍♀️")
    st.title("HR - Resume Screening Assistance")
    # st.subheader("I can help you in resume screening process")
    
    st.markdown("<div style='text-align: center;'>" "<h6>Please paste the 'Job Description' here</h3>" "</div>", unsafe_allow_html=True)
    job_description = st.text_area("",key="1")

    pdf = st.file_uploader("Upload 2 or more resumes here", type=["pdf"],accept_multiple_files=True)
    
    # document_count = st.text_input("Enter the number of 'RESUMES' to return",key="2")
    if pdf : 
        max_resumes = len(pdf) if pdf else 1
        document_count = st.selectbox("Select the number of 'RESUMES' to return", options=range(1, max_resumes))
    left_column, center_column, right_column = st.columns([5, 5, 5])
    with center_column:
        submit=st.button("Analyze the Resumes")
        
    if submit and pdf:
        with st.spinner('Processing the resumes'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])

            #Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

            #Create embeddings instance
            embeddings=create_embeddings_load_data()

            #Push data to PINECONE
            # push_to_pinecone(os.getenv("pinecone_api_key"),"gcp-starter","genai-doc",embeddings,final_docs_list)
            push_to_pinecone('pinecone_api_key',"gcp-starter","genai-doc",embeddings,final_docs_list)

            #Fecth relavant documents from PINECONE
            relavant_docs=similar_docs(job_description,document_count,'pinecone_api_key',"gcp-starter","genai-doc",embeddings,st.session_state['unique_id'])

            #t.write(relavant_docs)

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
                
                st.subheader("Resume ranking ➡️ "+str(item+1))

                #Displaying Filepath
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])

                #Introducing Expander feature
                with st.expander('Display the resume ✅'): 
                    st.info("**Match Score** : "+str(relavant_docs[item][1]))
                    #st.write("***"+relavant_docs[item][0].page_content)
                    
                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relavant_docs[item][0])
                    st.write("**Summary** : "+summary)
                    
                    
        left_column1, center_column1, right_column1 = st.columns([5, 8, 5])
        with center_column1:
            st.success("Hope I was able to save your time❤️")


#Invoking main function
if __name__ == '__main__':
    main()