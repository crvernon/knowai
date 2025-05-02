import fitz  # PyMuPDF
import os
import argparse # For command-line arguments
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import re
import concurrent.futures
import threading
_file_write_lock = threading.Lock()

from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError

# LangChain components

# Configure LangChain verbosity to avoid deprecated verbose import warning
from langchain.globals import set_verbose
set_verbose(False)

from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
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

# --- Neo4j Configuration ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "postgres")
NEO4J_DATABASE = "neo4j"

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
        raw_type = triple.get("type", "").strip()
        # Replace non-alphanumeric characters with underscore, collapse multiple underscores, and remove leading/trailing underscores
        relation_type = re.sub(r'[^0-9A-Za-z]', '_', raw_type)
        relation_type = re.sub(r'_+', '_', relation_type).strip('_').upper()

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

def build_knowledge_graph(pdf_path: str, output_cypher_file: str, write_output: bool = False):
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
    # Initialize document tracking
    doc_filename = os.path.basename(pdf_path)
    entities = set()

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
    # Initialize per-file chunk progress bar
    pbar = tqdm(total=len(chunks), desc=f"Processing {doc_filename}", unit="chunk")
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

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")

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
                validated_data = TripleList.model_validate(parsed_json)

                if validated_data and validated_data.triples:
                    # Convert Pydantic models to dicts using model_dump
                    chunk_triples = [t.model_dump() for t in validated_data.triples]
                    print(f"  Successfully parsed and validated {len(chunk_triples)} triples.")
                    total_triples += len(chunk_triples)
                    cypher_statements = generate_cypher(chunk_triples)
                    all_cypher_statements.extend(cypher_statements)
                    # Track entities for FROM_DOCUMENT relationships
                    entities.update([t['head'] for t in chunk_triples] + [t['tail'] for t in chunk_triples])
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

        # Update per-file progress
        pbar.update(1)

    pbar.close()
    print(f"\nExtraction complete. Total triples extracted: {total_triples}")

    # Add document node and link to each extracted entity
    all_cypher_statements.insert(0, f'MERGE (d:Document {{name: "{doc_filename}"}});')
    for entity in entities:
        safe_entity = entity.replace('"', '\\"')
        all_cypher_statements.append(
            f'MATCH (d:Document {{name: "{doc_filename}"}}), (n:Chunk {{name: "{safe_entity}"}}) '
            'MERGE (d)-[:FROM_DOCUMENT]->(n);'
        )

    # Optionally write Cypher statements to file
    if write_output and all_cypher_statements:
        try:
            with _file_write_lock:
                with open(output_cypher_file, 'a', encoding='utf-8') as f:
                    for stmt in all_cypher_statements:
                        f.write(stmt + "\n")
            print(f"Successfully wrote {len(all_cypher_statements)} Cypher statements to file {output_cypher_file}")
        except Exception as e:
            print(f"Error writing Cypher statements to file {output_cypher_file}: {e}")

    # 6. Execute Cypher Statements in Neo4j
    if all_cypher_statements:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            with driver.session(database=NEO4J_DATABASE) as session:
                for stmt in all_cypher_statements:
                    session.run(stmt)
            print(f"Successfully executed {len(all_cypher_statements)} Cypher statements in database {NEO4J_DATABASE}")
        except Exception as e:
            print(f"Error executing Cypher statements in Neo4j: {e}")
        finally:
            driver.close()
    else:
        print("No Cypher statements generated as no triples were extracted.")


# --- Command-Line Interface Functions ---

def expand_pdf_paths(paths: List[str]) -> List[str]:
    """
    Expand a list of files or directories into a flat list of PDF file paths.
    """
    pdf_list: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(".pdf"):
                    pdf_list.append(os.path.join(path, f))
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            pdf_list.append(path)
        else:
            print(f"Warning: {path} is not a valid PDF file or directory and will be skipped.")
    return pdf_list

def filter_existing_documents(pdf_list: List[str]) -> List[str]:
    """
    Return only those PDFs that do not yet have a Document node in Neo4j.
    """
    to_process: List[str] = []
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DATABASE) as session:
        for pdf in pdf_list:
            name = os.path.basename(pdf)
            record = session.run(
                "MATCH (d:Document {name: $name}) RETURN d",
                {"name": name}
            ).single()
            if record:
                print(f"Skipping {name}: already exists in graph.")
            else:
                to_process.append(pdf)
    driver.close()
    return to_process

def clear_output_file(output_file: str, write_file: bool) -> None:
    """
    Clear the output file if write_file is True.
    """
    if write_file and os.path.exists(output_file):
        os.remove(output_file)

def process_pdfs_parallel(pdf_list: List[str], output_file: str, write_file: bool) -> None:
    """
    Process a list of PDFs concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_pdf = {
            executor.submit(build_knowledge_graph, pdf, output_file, write_file): pdf
            for pdf in pdf_list
        }
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

def main() -> None:
    """
    Parse command-line arguments and orchestrate the PDF processing workflow.
    """
    parser = argparse.ArgumentParser(
        description="Build a Knowledge Graph from PDFs by extracting triples and generating Cypher statements."
    )
    parser.add_argument(
        "pdf_directory",
        nargs="+",
        help="Paths to the input PDF file(s) or directories."
    )
    parser.add_argument(
        "-o", "--output",
        default="knowledge_graph.cypher",
        help="Path to save the output Cypher file (default: knowledge_graph.cypher)."
    )
    parser.add_argument(
        "--write-file",
        action="store_true",
        help="Additionally write Cypher statements to the output file."
    )

    args = parser.parse_args()
    write_file = args.write_file

    # Prepare output file if needed
    clear_output_file(args.output, write_file)

    # Expand and filter PDFs
    pdf_list = expand_pdf_paths(args.pdf_directory)
    to_process = filter_existing_documents(pdf_list)

    if not to_process:
        print("No new PDFs to process.")
        return

    # Process new PDFs concurrently
    process_pdfs_parallel(to_process, args.output, write_file)

if __name__ == "__main__":
    main()
