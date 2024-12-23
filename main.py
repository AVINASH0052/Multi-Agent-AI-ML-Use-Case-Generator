# Import Required Libraries
import os
import requests
import kaggle
import streamlit as st
import graphviz
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
load_dotenv()

from huggingface_hub import login

# Log in to Hugging Face API
login("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")

NVIDIA_API_KEY = "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"  # Replace with your actual NVIDIA API key
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Load API Keys
KAGGLE_USERNAME = "avinashvikramsingh"
KAGGLE_KEY = "55fd683bd1a2e18e3b207bdce916976b"

# Research Agent using NVIDIA API
def research_industry(company_name: str):
    """Research the industry, segment, and strategic focus areas of a given company using NVIDIA API."""
    # Initial query to get the industry and strategic areas information
    initial_query = f"Research {company_name} company industry and segment it is working in. Identify key offerings and strategic areas like operations, supply chain, customer experience, etc."

    # Payload for initial NVIDIA API call
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

    # Make initial API request to NVIDIA
    response = requests.post(f"{NVIDIA_BASE_URL}/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        # Parse the response text from initial call
        response_data = response.json()
        response_text = response_data["choices"][0]["message"]["content"]
        
        # Refinement query to structure results into top three industries and strategic areas
        refine_query = (
            f"Based on the following information about {company_name}, create two lists: "
            "one named 'industry' containing the top three industry names and another named 'focus_areas' containing the top three strategic focus areas."
            f"\n\n{response_text}"
        )

        # Payload for refinement NVIDIA API call
        refine_payload = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [{"role": "user", "content": refine_query}],
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 1024,
        }

        # Make refinement API request to NVIDIA
        refine_response = requests.post(f"{NVIDIA_BASE_URL}/chat/completions", headers=headers, json=refine_payload)

        if refine_response.status_code == 200:
            refine_data = refine_response.json()
            refine_text = refine_data["choices"][0]["message"]["content"]
            
            # Extract industries and focus areas from structured response
            industries, focus_areas = [], []
            for line in refine_text.splitlines():
                if "industry" in line.lower():
                    industries = [item.strip() for item in line.replace("Industry:", "").split(",")][:3]
                elif "focus_areas" in line.lower():
                    focus_areas = [item.strip() for item in line.replace("Focus Areas:", "").split(",")][:3]
            
            return industries, focus_areas
        
        else:
            print(f"Error during refinement: {refine_response.status_code} - {refine_response.text}")
            return ["Industry Not Identified"], ["Focus Areas Not Identified"]
    
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ["Industry Not Identified"], ["Focus Areas Not Identified"]

# Market Standards & Use Case Generation Agent using NVIDIA API
def generate_use_cases(industry: list, focus_areas: list):
    """Generate AI/ML use cases for the given industry and focus areas using NVIDIA API."""
    if "Industry Not Identified" in industry or not focus_areas:
        return ["Unable to generate specific use cases due to lack of identified industry or focus areas."]
    
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
        use_cases = [line.lstrip('0123456789. ') for line in use_cases if line.strip()]  # Remove any leading numbers and dots
        return use_cases
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ["Error generating use cases with NVIDIA API."]

# Resource Asset Collection Agent
def collect_datasets(use_cases: list):
    """Collect relevant datasets for the generated use cases."""
    kaggle.api.authenticate()
    
    datasets = []
    for case in use_cases:
        try:
            results = kaggle.api.dataset_list(search=case, file_type="csv")
            limited_results = results[:5] if len(results) > 5 else results
            for result in limited_results:
                datasets.append((case, result.ref))
        except Exception as e:
            print(f"Error retrieving dataset for '{case}': {str(e)}")
            datasets.append((case, "Error retrieving dataset"))
    
    with open("datasets.md", "w") as f:
        for use_case, dataset_ref in datasets:
            dataset_link = f"https://www.kaggle.com/datasets/{dataset_ref}" if "Error" not in dataset_ref else "Dataset not available"
            f.write(f"### {use_case}\n* [{dataset_ref}]({dataset_link})\n\n")
    return datasets

def generate_final_proposal(industry: list, focus_areas: list, use_cases: list, datasets: list, company_name: str):
    """Generate a systematically organized final proposal document with streamlined content."""
    
    # Introduction with unified phrasing
    proposal_content = f"""
    # Final Proposal for {company_name}
    
    ## Industries: {', '.join(industry) if industry else 'Not Identified'}
    ## Focus Areas: {', '.join(focus_areas) if focus_areas else 'No specific focus areas identified'}
    
    ## Use Cases:
    Below are examples of AI/ML use cases in the industries of {', '.join(industry)} with a focus on {', '.join(focus_areas)}. Each use case includes potential applications, benefits, and challenges.
    
    """
    
    for case in use_cases:
        proposal_content += f"- {case.strip()}\n"

    # Datasets Section
    proposal_content += "\n\n## Datasets:\n"
    for case, dataset in datasets:
        dataset_link = f"https://www.kaggle.com/datasets/{dataset}" if "Error" not in dataset else "Dataset not available"
        proposal_content += f"- {case}: [{dataset_link}]\n"

    # Save to file
    with open("final_proposal.md", "w") as f:
        f.write(proposal_content)

# Architecture Flowchart Generation
def generate_architecture_flowchart():
    """Generate a flowchart representing the architecture of the system."""
    dot = graphviz.Digraph(comment='Multi-Agent Architecture')
    dot.node('A', 'Research Agent')
    dot.node('B', 'Market Standards & Use Case Generation Agent')
    dot.node('C', 'Resource Asset Collection Agent')
    dot.node('D', 'Final Proposal Generation')
    dot.node('E', 'Streamlit App')
    dot.edges([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
    dot.render('architecture_flowchart', format='png')

# Main Function
def main():
    st.title("Multi-Agent Architecture Demo")
    company_name = st.text_input("Enter Company Name")
    if company_name:
        industries, focus_areas = research_industry(company_name)
        use_cases = generate_use_cases(industries, focus_areas)
        datasets = collect_datasets(use_cases)
        generate_final_proposal(industries, focus_areas, use_cases, datasets, company_name)
        generate_architecture_flowchart()
        streamlit_app()

def streamlit_app():
    """Streamlit app to display the final proposal and architecture flowchart."""
    st.write("### Final Proposal")
    with open("final_proposal.md", "r") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
    st.image("architecture_flowchart.png")

if __name__ == "__main__":
    main()
