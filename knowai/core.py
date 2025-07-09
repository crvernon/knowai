# knowai/core.py
import logging
import os
from typing import List, Dict, Optional, Any, Callable

from dotenv import load_dotenv

from .agent import (
    GraphState,
    create_graph_app
)

from .errors import (
    RateLimitError,
    TokenLimitError
)

from knowai.agent import GLOBAL_PROGRESS_CB

K_CHUNKS_RETRIEVER_DEFAULT = 15
K_CHUNKS_RETRIEVER_ALL_DOCS_DEFAULT = 100000
MAX_CONVERSATION_TURNS_DEFAULT = 25
N_QUERY_ALTERNATIVES_DEFAULT = 1


logger = logging.getLogger(__name__)


def get_workflow_mermaid_diagram(save_to_file: Optional[str] = None) -> str:
    """
    Generate a Mermaid diagram representation of the KnowAI LangGraph workflow.

    This is a standalone function that creates a temporary graph instance
    to generate the diagram without needing a full KnowAIAgent instance.

    Parameters
    ----------
    save_to_file : Optional[str], default None
        If provided, save the Mermaid diagram to the specified file path.

    Returns
    -------
    str
        Mermaid diagram string that can be rendered in any Mermaid-compatible viewer.

    Examples
    --------
    >>> from knowai.core import get_workflow_mermaid_diagram
    >>> diagram = get_workflow_mermaid_diagram()
    >>> print(diagram)

    # Save to file
    >>> get_workflow_mermaid_diagram(save_to_file="workflow.md")
    """
    graph_app = create_graph_app()
    mermaid_diagram = graph_app.get_graph().draw_mermaid()

    if save_to_file:
        try:
            with open(save_to_file, 'w') as f:
                f.write("```mermaid\n")
                f.write(mermaid_diagram)
                f.write("\n```\n")
            logging.info(f"Mermaid diagram saved to: {save_to_file}")
        except Exception as e:
            logging.error(f"Failed to save Mermaid diagram to {save_to_file}: {e}")

    return mermaid_diagram


class KnowAIAgent:
    """
    Conversational Retrieval‑Augmented Generation (RAG) agent built on a
    LangGraph workflow.

    The agent owns a compiled LangGraph *graph_app* and a mutable
    ``session_state`` that flows through the graph. It exposes
    :pyfunc:`process_turn`, which takes the user's question plus optional
    UI parameters, executes the LangGraph asynchronously, updates
    conversation history, and returns structured results for display.

    Parameters
    ----------
    vectorstore_path : str
        Path on disk to the FAISS vector‑store directory.
    max_conversation_turns : int, default ``MAX_CONVERSATION_TURNS_DEFAULT``
        Number of past turns to retain in ``session_state``.
    k_chunks_retriever : int, default ``K_CHUNKS_RETRIEVER_DEFAULT``
        Top‑*k* chunks returned by the base retriever when no re‑ranking
        is applied.
    env_file_path : Optional[str], default ``None``
        Explicit path to a *.env* file containing Azure/OpenAI settings.
        If ``None``, the constructor attempts auto‑detection.
    initial_state_overrides : Optional[Dict[str, Any]], default ``None``
        Mapping of ``GraphState`` keys to override their default initial
        values. Unknown keys are ignored with a warning.
    log_graph : bool, default ``False``
        Whether to log the Mermaid diagram of the workflow graph.

    process_files_individually : bool, default ``False``
        Whether to process each file individually and then consolidate responses.
        When True, each file is processed separately by the LLM and then all
        responses are combined into a final answer. When False, uses traditional
        batch processing approach.

    Attributes
    ----------
    graph_app : langgraph.Graph
        Compiled LangGraph responsible for end‑to‑end RAG processing.
    session_state : GraphState
        Mutable state object passed into each LangGraph invocation.
    max_conversation_turns : int
        Maximum number of turns stored in ``conversation_history``.
    """
    def __init__(
        self,
        vectorstore_path: str,
        max_conversation_turns: int = MAX_CONVERSATION_TURNS_DEFAULT,
        k_chunks_retriever: int = K_CHUNKS_RETRIEVER_DEFAULT,
        k_chunks_retriever_all_docs: int = K_CHUNKS_RETRIEVER_ALL_DOCS_DEFAULT,
        env_file_path: Optional[str] = None,
        initial_state_overrides: Optional[Dict[str, Any]] = None,
        log_graph: bool = False,

        process_files_individually: bool = True
    ) -> None:
        if env_file_path and os.path.exists(env_file_path):
            load_dotenv(dotenv_path=env_file_path)
            logging.info(f"Loaded environment variables from: {env_file_path}")
        elif load_dotenv():  # Try to auto-detect .env
            logging.info("Loaded environment variables from a .env file.")
        else:
            logging.warning(
                "No .env file explicitly provided or auto-detected. "
                "Ensure environment variables are set."
            )

        self.graph_app = create_graph_app()
        self.max_conversation_turns = max_conversation_turns

        self.session_state: GraphState = {
            "embeddings": None,
            "vectorstore_path": vectorstore_path,
            "vectorstore": None,
            "llm_large": None,
            "llm_small": None,
            "retriever": None,
            "allowed_files": None,
            "question": None,
            "documents_by_file": None,
            "n_alternatives": N_QUERY_ALTERNATIVES_DEFAULT,
            "k_per_query": 10,
            "generation": None,
            "conversation_history": [],
            "raw_documents_for_synthesis": None,
            "combined_documents": None,
            "detailed_response_desired": True,
            "k_chunks_retriever": k_chunks_retriever,
            "k_chunks_retriever_all_docs": k_chunks_retriever_all_docs,
            "generated_queries": None,
            "query_embeddings": None,
            "streaming_callback": None,
            "__progress_cb__": None,
            "max_tokens_per_batch": int(1_000_000 * 0.9),  # GPT-4.1 with 10% safety margin
            "batch_results": None,

            "process_files_individually": process_files_individually,
            "individual_file_responses": None,
            "hierarchical_consolidation_results": None,
            "show_detailed_individual_responses": False,
            "detailed_responses_for_ui": None,
            "rate_limit_error_occurred": False,
        }

        if initial_state_overrides:
            for key, value in initial_state_overrides.items():
                if key in self.session_state:
                    self.session_state[key] = value  # type: ignore
                else:
                    logging.warning(
                        f"Ignoring unknown key '{key}' in initial_state_overrides."
                    )

        if log_graph:
            logging.info(self.graph_app.get_graph().draw_mermaid())

        logging.info(
            "KnowAIAgent initialized. Component loading will occur on the first "
            "'process_turn' call."
        )

    def get_graph_mermaid(self, save_to_file: Optional[str] = None) -> str:
        """
        Generate a Mermaid diagram representation of the LangGraph workflow.

        Parameters
        ----------
        save_to_file : Optional[str], default None
            If provided, save the Mermaid diagram to the specified file path.

        Returns
        -------
        str
            Mermaid diagram string that can be rendered in any Mermaid-compatible viewer.

        Examples
        --------
        >>> agent = KnowAIAgent("path/to/vectorstore")
        >>> mermaid_diagram = agent.get_graph_mermaid()
        >>> print(mermaid_diagram)

        # Save to file
        >>> agent.get_graph_mermaid(save_to_file="workflow_diagram.md")
        """
        mermaid_diagram = self.graph_app.get_graph().draw_mermaid()

        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write("```mermaid\n")
                    f.write(mermaid_diagram)
                    f.write("\n```\n")
                logging.info(f"Mermaid diagram saved to: {save_to_file}")
            except Exception as e:
                logging.error(f"Failed to save Mermaid diagram to {save_to_file}: {e}")

        return mermaid_diagram

    async def process_turn(
        self,
        user_question: Optional[str] = None,
        selected_files: Optional[List[str]] = None,
        n_alternatives_override: Optional[int] = None,
        k_per_query_override: Optional[int] = None,
        progress_cb: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        detailed_response_desired: Optional[bool] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        process_files_individually: Optional[bool] = None,
        show_detailed_individual_responses: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Processes a single conversational turn.

        Parameters
        ----------
        user_question : Optional[str]
            The user's question to process.
        selected_files : Optional[List[str]]
            List of files to search in.
        n_alternatives_override : Optional[int]
            Override for number of alternative queries.
        k_per_query_override : Optional[int]
            Override for chunks per query.
        progress_cb : Optional[Callable[[str, str, Dict[str, Any]], None]]
            Progress callback for node-level updates.
        detailed_response_desired : Optional[bool]
            Whether to use detailed (large) or simple (small) LLM.
        streaming_callback : Optional[Callable[[str], None]]
            Callback function to stream tokens as they're generated.
            If provided, the final LLM response will be streamed in real-time.
        process_files_individually : Optional[bool]
            Whether to process files individually instead of as a batch.
            When True, each file is processed separately by the LLM and then
            all responses are combined into a final answer. When False, uses
            traditional batch processing approach.
        show_detailed_individual_responses : Optional[bool]
            Whether to include individual document responses in the final output.
            When True, the response will include a section with detailed responses
            from each individual document after the main summary.

        Returns:
            A dictionary containing:
                "generation": The final assistant response string.
                "documents_by_file": Dictionary of retrieved documents per file.
                "raw_documents_for_synthesis": Formatted raw documents.
        """
        self.session_state["question"] = user_question
        self.session_state["allowed_files"] = selected_files
        self.session_state["__progress_cb__"] = progress_cb
        self.session_state["streaming_callback"] = streaming_callback

        if detailed_response_desired is not None:
            self.session_state["detailed_response_desired"] = detailed_response_desired

        if n_alternatives_override is not None:
            self.session_state["n_alternatives"] = n_alternatives_override
        if k_per_query_override is not None:
            self.session_state["k_per_query"] = k_per_query_override

        if process_files_individually is not None:
            self.session_state["process_files_individually"] = process_files_individually

        if show_detailed_individual_responses is not None:
            self.session_state["show_detailed_individual_responses"] = show_detailed_individual_responses

        # Ensure all required GraphState keys are present
        for key in GraphState.__annotations__.keys():
            if key not in self.session_state:
                # Set defaults for any missing keys to ensure GraphState is complete
                if key == "conversation_history":
                    self.session_state[key] = []  # type: ignore
                elif key == "k_chunks_retriever":
                    self.session_state[key] = K_CHUNKS_RETRIEVER_DEFAULT  # type: ignore
                else:
                    self.session_state[key] = None  # type: ignore

        # Clear previous RAG artifacts for new questions
        if user_question:
            self.session_state["documents_by_file"] = None
            self.session_state["raw_documents_for_synthesis"] = None
            self.session_state["batch_results"] = None

        # Set global progress callback for workflow (must set on the *agent* module
        # not just this local alias, otherwise the graph nodes will not see it)
        import knowai.agent as _agent_mod  # late import to avoid circular issues

        _agent_mod.GLOBAL_PROGRESS_CB = progress_cb
        try:
            updated_state = await self.graph_app.ainvoke(self.session_state)  # type: ignore
        except RateLimitError as e:
            # Rate limit error - stop workflow and return error message
            logging.warning(f"Rate limit error caught in process_turn: {e}")
            return {
                "generation": str(e),
                "documents_by_file": None,
                "raw_documents_for_synthesis": None,
                "detailed_responses": None,
            }
        except TokenLimitError as e:
            # Token limit error - stop workflow and return error message
            logging.error(f"Token limit error caught in process_turn: {e}")
            return {
                "generation": str(e),
                "documents_by_file": None,
                "raw_documents_for_synthesis": None,
                "detailed_responses": None,
            }
        finally:
            # Clear the callback on the agent module so it does not leak
            _agent_mod.GLOBAL_PROGRESS_CB = None
        self.session_state.update(updated_state)  # type: ignore
        
        # Ensure detailed responses are properly transferred from workflow state
        if "detailed_responses_for_ui" in updated_state:
            self.session_state["detailed_responses_for_ui"] = updated_state["detailed_responses_for_ui"]

        assistant_response_str = self.session_state.get(
            "generation", "I'm sorry, I couldn't formulate a response."
        )

        # Ensure assistant_response_str is a string
        if assistant_response_str is None:
            assistant_response_str = (
                "I'm sorry, I couldn't formulate a response based on the "
                "provided information."
            )

        # Update conversation history
        if user_question and assistant_response_str:
            current_history = self.session_state.get("conversation_history")
            if current_history is None:
                current_history = []

            current_history.append({
                "user_question": user_question,
                "assistant_response": assistant_response_str
            })
            self.session_state["conversation_history"] = current_history[-self.max_conversation_turns:]
            logging.info(
                f"Conversation history updated. New length: "
                f"{len(self.session_state['conversation_history'])}"
            )

        # Ensure detailed responses are properly transferred from workflow state
        if "detailed_responses_for_ui" in updated_state:
            self.session_state["detailed_responses_for_ui"] = updated_state["detailed_responses_for_ui"]

        # Get detailed responses from state if they exist
        detailed_responses = self.session_state.get("detailed_responses_for_ui")

        # Return a dictionary with all necessary info for the UI
        return {
            "generation": assistant_response_str,
            "documents_by_file": self.session_state.get("documents_by_file"),
            "raw_documents_for_synthesis": self.session_state.get("raw_documents_for_synthesis"),
            "detailed_responses": detailed_responses,
        }
