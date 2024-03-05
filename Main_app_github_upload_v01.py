import time
import openai
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the OpenAI client with your API key
openai.api_key = "YOUR_API_KEY_FROM_OPENAI"

def simulate_processing():
    time.sleep(2)

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def calculate_field_similarity(resume_text, fields):
    scores = {}

    # Generate job descriptions for each field and level
    generated_descriptions = generate_job_descriptions(list(fields.keys()), ["entry-level", "mid-senior level", "advanced level"])

    for field_level, job_description in generated_descriptions.items():
        similarity_score = calculate_similarity(resume_text, job_description)
        field, level = field_level.split(" - ")
        scores.setdefault(field, {})[level] = similarity_score

    return scores


def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def suggest_technical_keywords_gpt3(job_description):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Suggest technical keywords for a resume based on the following job description:\n\n{job_description}"}
        ],
        temperature=0.7,
        max_tokens=150,
        n=5
    )

    suggestions = [response.choices[0].message.content.strip()]
    return suggestions

def suggest_keywords_for_resume_improvement_gpt3(resume_text, job_description):
    job_keywords = suggest_technical_keywords_gpt3(job_description)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract technical keywords from the following resume:\n\n{resume_text}"}
        ],
        temperature=0.7,
        max_tokens=150,
        n=5
    )

    resume_keywords = [choice.message.content.strip() for choice in response.choices]
    missing_keywords = list(set(job_keywords) - set(resume_keywords))
    return missing_keywords


def generate_job_descriptions(fields, levels):
    descriptions = {}
    for field in fields:
        for level in levels:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate a job description for an {level} {field} position."}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                n=1
            )
            description = response.choices[0].message.content.strip()
            descriptions[f"{field} - {level}"] = description
    return descriptions

def main():
    """Main function to run the Streamlit app."""
    st.title("ATS-Friendly Resume Matcher")

    uploaded_resume = st.file_uploader("Upload Your Resume (PDF or Text)", type=["pdf", "txt"])
    job_description = st.text_area("Input the Job Description Here")

    if st.button("Power Analyze"):
        if uploaded_resume:
            with st.spinner("Uploading and Analyzing Resume..."):
                # Simulate processing time
                simulate_processing()

                # Read the content of the uploaded resume
                resume_text = ""
                if uploaded_resume.type == "text/plain":
                    resume_text = uploaded_resume.read()
                elif uploaded_resume.type == "application/pdf":
                    # Extract text from the PDF file
                    resume_text = extract_text_from_pdf(uploaded_resume)
                else:
                    st.error("Unsupported file type. Please upload a plain text or PDF resume.")

                # Calculate the similarity score
                if resume_text:
                    # Display a spinner during analysis
                    with st.spinner("Analyzing Resume..."):
                        # Simulate processing time
                        simulate_processing()

                        # Display the results
                        st.success("General Analysis complete!")

                        # Display predefined field-specific similarity scores for general analysis
                        predefined_fields = {
                            "Information Technology": "Your IT job description here",
                            "Computer Science": "Your Computer Science job description here",
                            "Data Science": "Your Data Science job description here",
                            "Machine Learning": "Your Machine Learning job description here",
                            "Business Analysis": "Your Business Analysis job description here",
                            "Cloud Computing": "Your Cloud Computing job description here",
                            # Add more predefined fields as needed
                        }
                        field_scores = calculate_field_similarity(resume_text, predefined_fields)

                        # Organize the similarity scores in a DataFrame
                        df_field_scores = pd.DataFrame(field_scores).T
                        st.table(df_field_scores.style.format("{:.2%}"))

                        # Add your specific output for general analysis here
                        st.info("Placeholder for general analysis output.")
                else:
                    st.error("Error processing the resume. Please upload a valid file.")
        else:
            st.info("Upload your resume before analyzing.")


    if st.button("Matching Analyze"):
        if uploaded_resume and job_description:
            with st.spinner("Uploading and Combining Analyzing Resume..."):
                simulate_processing()
                resume_text = ""
                if uploaded_resume.type == "text/plain":
                    resume_text = uploaded_resume.getvalue().decode("utf-8")
                elif uploaded_resume.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_resume)

                if resume_text:
                    similarity_score = calculate_similarity(resume_text, job_description)
                    st.success(f"Combining Analysis complete! Similarity Score: {similarity_score:.2%}")
                    missing_keywords = suggest_keywords_for_resume_improvement_gpt3(resume_text, job_description)
                    if missing_keywords:
                        # df_keywords = pd.DataFrame({"Missing Keywords": missing_keywords})
                        st.text(missing_keywords[0])
                    else:
                        st.info("No suggested keywords for resume improvement.")
                else:
                    st.error("Error processing the resume. Please upload a valid file.")
        else:
            st.info("Upload both your resume and input the job description before analyzing.")

if __name__ == "__main__":
    main()
import time
import openai
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the OpenAI client with your API key
openai.api_key = "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"

def simulate_processing():
    time.sleep(2)

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def calculate_field_similarity(resume_text, fields):
    scores = {}

    # Generate job descriptions for each field and level
    generated_descriptions = generate_job_descriptions(list(fields.keys()), ["entry-level", "mid-senior level", "advanced level"])

    for field_level, job_description in generated_descriptions.items():
        similarity_score = calculate_similarity(resume_text, job_description)
        field, level = field_level.split(" - ")
        scores.setdefault(field, {})[level] = similarity_score

    return scores


def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def suggest_technical_keywords_gpt3(job_description):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Suggest technical keywords for a resume based on the following job description:\n\n{job_description}"}
        ],
        temperature=0.7,
        max_tokens=150,
        n=5
    )

    suggestions = [response.choices[0].message.content.strip()]
    return suggestions

def suggest_keywords_for_resume_improvement_gpt3(resume_text, job_description):
    job_keywords = suggest_technical_keywords_gpt3(job_description)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract technical keywords from the following resume:\n\n{resume_text}"}
        ],
        temperature=0.7,
        max_tokens=150,
        n=5
    )

    resume_keywords = [choice.message.content.strip() for choice in response.choices]
    missing_keywords = list(set(job_keywords) - set(resume_keywords))
    return missing_keywords


def generate_job_descriptions(fields, levels):
    descriptions = {}
    for field in fields:
        for level in levels:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate a job description for an {level} {field} position."}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                n=1
            )
            description = response.choices[0].message.content.strip()
            descriptions[f"{field} - {level}"] = description
    return descriptions

def main():
    """Main function to run the Streamlit app."""
    st.title("ATS-Friendly Resume Matcher")

    uploaded_resume = st.file_uploader("Upload Your Resume (PDF or Text)", type=["pdf", "txt"])
    job_description = st.text_area("Input the Job Description Here")

    if st.button("Power Analyze"):
        if uploaded_resume:
            with st.spinner("Uploading and Analyzing Resume..."):
                # Simulate processing time
                simulate_processing()

                # Read the content of the uploaded resume
                resume_text = ""
                if uploaded_resume.type == "text/plain":
                    resume_text = uploaded_resume.read()
                elif uploaded_resume.type == "application/pdf":
                    # Extract text from the PDF file
                    resume_text = extract_text_from_pdf(uploaded_resume)
                else:
                    st.error("Unsupported file type. Please upload a plain text or PDF resume.")

                # Calculate the similarity score
                if resume_text:
                    # Display a spinner during analysis
                    with st.spinner("Analyzing Resume..."):
                        # Simulate processing time
                        simulate_processing()

                        # Display the results
                        st.success("General Analysis complete!")

                        # Display predefined field-specific similarity scores for general analysis
                        predefined_fields = {
                            "Information Technology": "Your IT job description here",
                            "Computer Science": "Your Computer Science job description here",
                            "Data Science": "Your Data Science job description here",
                            "Machine Learning": "Your Machine Learning job description here",
                            "Business Analysis": "Your Business Analysis job description here",
                            "Cloud Computing": "Your Cloud Computing job description here",
                            # Add more predefined fields as needed
                        }
                        field_scores = calculate_field_similarity(resume_text, predefined_fields)

                        # Organize the similarity scores in a DataFrame
                        df_field_scores = pd.DataFrame(field_scores).T
                        st.table(df_field_scores.style.format("{:.2%}"))

                        # Add your specific output for general analysis here
                        st.info("Placeholder for general analysis output.")
                else:
                    st.error("Error processing the resume. Please upload a valid file.")
        else:
            st.info("Upload your resume before analyzing.")


    if st.button("Matching Analyze"):
        if uploaded_resume and job_description:
            with st.spinner("Uploading and Combining Analyzing Resume..."):
                simulate_processing()
                resume_text = ""
                if uploaded_resume.type == "text/plain":
                    resume_text = uploaded_resume.getvalue().decode("utf-8")
                elif uploaded_resume.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_resume)

                if resume_text:
                    similarity_score = calculate_similarity(resume_text, job_description)
                    st.success(f"Combining Analysis complete! Similarity Score: {similarity_score:.2%}")
                    missing_keywords = suggest_keywords_for_resume_improvement_gpt3(resume_text, job_description)
                    if missing_keywords:
                        # df_keywords = pd.DataFrame({"Missing Keywords": missing_keywords})
                        st.text(missing_keywords[0])
                    else:
                        st.info("No suggested keywords for resume improvement.")
                else:
                    st.error("Error processing the resume. Please upload a valid file.")
        else:
            st.info("Upload both your resume and input the job description before analyzing.")

if __name__ == "__main__":
    main()
