# Resume Assistance



https://github.com/yadav-Simran/HR-RESUME-SCREENING-LLM-MODEL/assets/123407453/3a16d7b1-9c4a-48fb-be98-fb41aabdf4ca



# Resume Screening Assistance App

This application streamlines the resume screening process for HR professionals using the power of AI and Langchain. The power of vectorDB is commendable. used Pinecone vector database.

## Features:
- **Job Description Analysis:** Paste the job description to identify key skills and requirements.
- **Resume Upload & Processing:** Upload multiple resumes in PDF format. The app extracts text and creates document embeddings for analysis.
- **AI-Powered Matching:** The app uses Pinecone, a vector database, to efficiently find the most relevant resumes based on the job description.
- **Ranked Results:** The app presents a ranked list of resumes along with their match scores.
- **Resume Summaries:** Expand each resume to view a concise summary generated by a large language model (LLM), providing a quick overview of the candidate's qualifications.
- **Efficient Screening:** Saves HR professionals significant time by quickly identifying the most promising candidates.

## Technologies Used:
- Streamlit: For building the user interface.
- Langchain: Framework for developing applications powered by language models.
- OpenAI Embeddings: Generating text embeddings for semantic search.
- Pinecone: Vector database for efficient storage and retrieval of embeddings.
- PyPDF: Extracting text from PDF files.
- HuggingFaceHub (Optional): Alternative LLM provider.

## Installation and Setup:
1. Clone the repository: `git clone https://github.com/yadav-Simran/HR-RESUME-SCREENING-LLM-MODEL.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Pinecone:
   - Create a Pinecone account and obtain your API key and environment details.
   - Update the `.env` file with your Pinecone credentials and OpenAI API key (if using OpenAI embeddings).
4. Run the app: `streamlit run app.py`

## Usage:
1. Paste the job description into the text area.
2. Upload PDF resumes using the file uploader.
3. Select the desired number of resumes to return.
4. Click the "Analyze the Resumes" button.
5. Review the ranked results and summaries to identify the best candidates.

## Customization:
- **LLM Provider:** Choose between OpenAI or HuggingFaceHub for generating summaries.
- **Number of Results:** Adjust the number of resumes returned based on your needs.
- **Embeddings Model:** Experiment with different embedding models for potentially improved accuracy.

## Contributing:
Contributions are welcome! Feel free to submit pull requests for bug fixes, enhancements, or new features.

.

