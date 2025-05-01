import streamlit as st # Using streamlit temporarily for status updates if run via streamlit run
import fitz  # PyMuPDF
import os
import argparse # For command-line arguments
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# LangChain components
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.output_parsers import StrOutputParser # Using string output parser now

# --- Azure OpenAI Configuration ---
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY", default=None)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", default=None)
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# --- Constants ---
CHUNK_SIZE = 1500 # Process slightly larger chunks for KG extraction
CHUNK_OVERLAP = 200

# --- Pydantic Schemas for LLM Output Validation ---

class KnowledgeTriple(BaseModel):
    """Represents a single relationship triple identified in the text."""
    head: str = Field(description="The subject entity (node) of the relationship. Should be a noun or noun phrase.")
    type: str = Field(description="The type of the relationship (edge) connecting the head and tail entities. Should be a verb or verb phrase.")
    tail: str = Field(description="The object entity (node) of the relationship. Should be a noun or noun phrase.")

class TripleList(BaseModel):
    """A list of knowledge triples extracted from a text chunk."""
    triples: List[KnowledgeTriple]

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extracts all text content from a PDF file."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n" # Add newline between pages
        doc.close()
        print(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def generate_cypher(triples: List[Dict[str, Any]], node_label: str = "Chunk") -> List[str]:
    """
    Generates Neo4j Cypher MERGE statements from extracted triples.

    Args:
        triples: A list of dictionaries, where each dict represents a triple
                 (e.g., {"head": "X", "type": "REL", "tail": "Y"}).
        node_label: The label to use for nodes in the graph (default: "Chunk").

    Returns:
        A list of Cypher MERGE statements.
    """
    cypher_statements = []
    for triple in triples:
        # Basic cleaning/escaping could be added here if needed
        head_entity = triple.get("head", "").strip()
        tail_entity = triple.get("tail", "").strip()
        relation_type = triple.get("type", "").strip().replace(" ", "_").upper() # Format relationship type

        if not head_entity or not tail_entity or not relation_type:
            print(f"Skipping invalid triple: {triple}")
            continue

        # Use MERGE to avoid creating duplicate nodes and relationships
        # Creates nodes with a 'name' property and the specified label
        # Creates relationships with the extracted type
        # Escape quotes within entity names for Cypher compatibility
        head_entity_cypher = head_entity.replace('"', '\\"')
        tail_entity_cypher = tail_entity.replace('"', '\\"')

        cypher = (
            f'MERGE (h:{node_label} {{name: "{head_entity_cypher}"}}) '
            f'MERGE (t:{node_label} {{name: "{tail_entity_cypher}"}}) '
            f'MERGE (h)-[:{relation_type}]->(t)'
        )
        cypher_statements.append(cypher + ";") # Add semicolon for Neo4j Browser/cypher-shell
    return cypher_statements

# --- Main Processing Function ---

def build_knowledge_graph(pdf_path: str, output_cypher_file: str):
    """
    Extracts knowledge triples from a PDF and saves them as Cypher statements.

    Args:
        pdf_path: Path to the input PDF file.
        output_cypher_file: Path to save the generated Cypher statements.
    """
    if not api_key or not azure_endpoint:
        print("Error: Azure OpenAI credentials (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT) not found in environment.")
        return

    print(f"Starting knowledge graph extraction for: {pdf_path}")

    # 1. Extract Text
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(pdf_text)
    print(f"Split text into {len(chunks)} chunks.")
    if not chunks:
        print("Error: No text chunks generated.")
        return

    # 3. Setup LLM for Triple Extraction (without structured output)
    try:
        print(f"Initializing AzureChatOpenAI (Deployment: {deployment})...")
        llm = AzureChatOpenAI(
            temperature=0.0, # Low temp for deterministic extraction
            api_key=api_key,
            openai_api_version=openai_api_version,
            azure_deployment=deployment,
            azure_endpoint=azure_endpoint,
            max_tokens=1024 # Allow sufficient tokens for potentially many triples
        )
        print("LLM initialized.")
    except Exception as e:
        print(f"Error initializing Azure LLM: {e}")
        return

    # 4. Define Extraction Prompt (asking for JSON output)
    extraction_prompt = ChatPromptTemplate.from_template(
        """You are an expert knowledge graph engineer. Your task is to extract structured information
        from the following text chunk extracted from a technical report. Identify key entities (nouns or noun phrases)
        and the relationships (verbs or verb phrases) connecting them. Focus on clear, factual relationships.

        Represent the extracted information as a JSON object containing a single key "triples", which is a list of objects.
        Each object in the "triples" list should represent a single relationship and have the keys "head", "type", and "tail".

        Example Output Format:
        ```json
        {{
          "triples": [
            {{"head": "Subject Entity 1", "type": "RELATIONSHIP_TYPE", "tail": "Object Entity 1"}},
            {{"head": "Subject Entity 2", "type": "ANOTHER_RELATIONSHIP", "tail": "Object Entity 2"}}
          ]
        }}
        ```

        Rules:
        - Extract only relationships explicitly stated or strongly implied in the text.
        - Be concise and consistent in naming entities and relationship types.
        - Avoid extracting overly generic or trivial relationships.
        - If no significant relationships are found in the chunk, return a JSON object with an empty "triples" list: `{{"triples": []}}`.
        - Ensure the output is a valid JSON object adhering strictly to the specified format.

        Text Chunk:
        ```{text_chunk}```

        Output (Valid JSON object following the specified format):"""
    )

    # 5. Process Chunks and Generate Cypher
    all_cypher_statements = []
    total_triples = 0

    # Use st.progress if running via streamlit, otherwise just print
    progress_bar = None
    try:
        # Check if running within Streamlit before attempting to use st.progress
        get_ipython # Simple check if running in IPython/Jupyter like env
        # If not in streamlit, don't use progress bar
    except NameError:
        try:
            # Attempt to use st.progress, will fail if not in streamlit context
             progress_bar = st.progress(0.0)
        except Exception:
            progress_bar = None # Not running in streamlit

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
        if progress_bar:
            progress_bar.progress(float(i+1)/len(chunks), text=f"Processing chunk {i + 1}/{len(chunks)}")

        try:
            # Create chain and invoke (LLM output is now a string)
            chain = extraction_prompt | llm | StrOutputParser()
            llm_output_str: str = chain.invoke({"text_chunk": chunk})

            # Attempt to parse the string output as JSON
            try:
                # Clean potential markdown code fences
                if llm_output_str.startswith("```json"):
                    llm_output_str = llm_output_str[7:]
                if llm_output_str.endswith("```"):
                    llm_output_str = llm_output_str[:-3]
                llm_output_str = llm_output_str.strip()

                # Parse JSON
                parsed_json = json.loads(llm_output_str)

                # Validate parsed JSON against Pydantic model
                validated_data = TripleList.parse_obj(parsed_json)

                if validated_data and validated_data.triples:
                    chunk_triples = [t.dict() for t in validated_data.triples] # Convert Pydantic models to dicts
                    print(f"  Successfully parsed and validated {len(chunk_triples)} triples.")
                    total_triples += len(chunk_triples)
                    cypher_statements = generate_cypher(chunk_triples)
                    all_cypher_statements.extend(cypher_statements)
                else:
                    print("  No triples extracted or validated from this chunk.")

            except json.JSONDecodeError as json_err:
                print(f"  Error: Failed to decode LLM output as JSON. Error: {json_err}")
                print(f"  LLM Output String: {llm_output_str}")
            except ValidationError as val_err:
                print(f"  Error: LLM output JSON does not match expected schema. Error: {val_err}")
                print(f"  Parsed JSON: {parsed_json}")
            except Exception as parse_err:
                 print(f"  Error parsing or validating LLM output: {parse_err}")
                 print(f"  LLM Output String: {llm_output_str}")


        except Exception as e:
            print(f"  Error invoking LLM for chunk {i + 1}: {e}")
            # Optionally add retry logic here
            continue # Move to the next chunk

    if progress_bar:
        progress_bar.progress(1.0, text="Processing complete.")

    print(f"\nExtraction complete. Total triples extracted: {total_triples}")

    # 6. Save Cypher Statements
    if all_cypher_statements:
        try:
            with open(output_cypher_file, 'w', encoding='utf-8') as f:
                for stmt in all_cypher_statements:
                    f.write(stmt + "\n")
            print(f"Successfully saved {len(all_cypher_statements)} Cypher statements to {output_cypher_file}")
        except Exception as e:
            print(f"Error writing Cypher file {output_cypher_file}: {e}")
    else:
        print("No Cypher statements generated as no triples were extracted.")

# --- Command-Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Knowledge Graph from a PDF by extracting triples and generating Cypher statements.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", default="knowledge_graph.cypher", help="Path to save the output Cypher file (default: knowledge_graph.cypher).")

    args = parser.parse_args()

    build_knowledge_graph(args.pdf_path, args.output)
