# Import Required Libraries
import os
import requests
import streamlit as st
import graphviz
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import login

# Load Environment Variables
load_dotenv()

# Set Kaggle API credentials using environment variables
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "avinashvikramsingh")  # Replace with your username if needed
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "55fd683bd1a2e18e3b207bdce916976b")  # Replace with your API key if needed

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Log in to Hugging Face API
login(os.getenv("HF_API_KEY", "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"))

# NVIDIA API Key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")  # Replace with your NVIDIA API key
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Research Agent using NVIDIA API
def research_industry(company_name: str):
    """Research the industry, segment, and strategic focus areas of a given company using NVIDIA API."""
    initial_query = f"Research {company_name} company industry and segment it is working in. Identify key offerings and strategic areas."

    payload = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [{"role": "user", "content": initial_query}],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(f"{NVIDIA_BASE_URL}/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        response_text = response_data["choices"][0]["message"]["content"]

        refine_query = (
            f"Based on the following information about {company_name}, create two lists: "
            "one named 'industry' containing the top three industry names and another named 'focus_areas' containing the top three strategic focus areas."
            f"\n\n{response_text}"
        )

        refine_payload = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [{"role": "user", "content": refine_query}],
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 1024,
        }

        refine_response = requests.post(f"{NVIDIA_BASE_URL}/chat/completions", headers=headers, json=refine_payload)

        if refine_response.status_code == 200:
            refine_data = refine_response.json()
            refine_text = refine_data["choices"][0]["message"]["content"]
            industries, focus_areas = [], []

            for line in refine_text.splitlines():
                if "industry" in line.lower():
                    industries = [item.strip() for item in line.replace("Industry:", "").split(",")][:3]
                elif "focus_areas" in line.lower():
                    focus_areas = [item.strip() for item in line.replace("Focus Areas:", "").split(",")][:3]

            return industries, focus_areas
        else:
            return ["Industry Not Identified"], ["Focus Areas Not Identified"]
    else:
        return ["Industry Not Identified"], ["Focus Areas Not Identified"]

# Generate AI/ML Use Cases
def generate_use_cases(industry: list, focus_areas: list):
    """Generate AI/ML use cases for the given industry and focus areas using NVIDIA API."""
    industry_text = ", ".join(industry)
    focus_text = ", ".join(focus_areas)
    template_text = (
        f"Provide examples of AI/ML use cases for the {industry_text} industries with a focus on {focus_text}. "
        "List how AI can be applied in each area, including potential benefits and challenges."
    )

    payload = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "messages": [{"role": "user", "content": template_text}],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(f"{NVIDIA_BASE_URL}/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        use_cases = response_data["choices"][0]["message"]["content"].split("\n")
        return [case.strip() for case in use_cases if case.strip()]
    else:
        return ["Error generating use cases with NVIDIA API."]

# Collect Datasets from Kaggle
def collect_datasets(use_cases: list):
    """Collect relevant datasets for the generated use cases."""
    datasets = []
    for case in use_cases:
        try:
            results = api.dataset_list(search=case, file_type="csv")
            datasets.extend([(case, result.ref) for result in results[:5]])
        except Exception as e:
            datasets.append((case, f"Error retrieving dataset: {e}"))

    return datasets

# Generate Final Proposal
def generate_final_proposal(industry, focus_areas, use_cases, datasets, company_name):
    """Generate a systematic final proposal document."""
    proposal_content = f"""
    # Final Proposal for {company_name}

    ## Industries
    {', '.join(industry)}

    ## Focus Areas
    {', '.join(focus_areas)}

    ## Use Cases
    {', '.join(use_cases)}

    ## Datasets
    {', '.join([f"{case}: {ref}" for case, ref in datasets])}
    """

    with open("final_proposal.md", "w") as f:
        f.write(proposal_content)

# Main Function for Streamlit App
def main():
    st.title("Multi-Agent AI/ML Use Case Generator")
    
    # Input field for company name
    company_name = st.text_input("Enter Company Name")
    
    if company_name:
        # Generate industries and focus areas
        industries, focus_areas = research_industry(company_name)
        
        # Display industries and focus areas on the page
        st.write("### Identified Industries:")
        st.write(", ".join(industries))
        
        st.write("### Identified Focus Areas:")
        st.write(", ".join(focus_areas))
        
        # Generate use cases
        use_cases = generate_use_cases(industries, focus_areas)
        
        # Display use cases on the page
        st.write("### Generated Use Cases:")
        for idx, case in enumerate(use_cases, 1):
            st.write(f"{idx}. {case}")
        
        # Collect datasets
        datasets = collect_datasets(use_cases)
        
        # Display datasets on the page
        st.write("### Relevant Datasets:")
        for case, dataset_ref in datasets:
            if "Error" in dataset_ref:
                st.write(f"Use Case: {case} - Dataset Not Available")
            else:
                st.write(f"Use Case: {case} - [Dataset Link](https://www.kaggle.com/datasets/{dataset_ref})")

if __name__ == "__main__":
    main()
