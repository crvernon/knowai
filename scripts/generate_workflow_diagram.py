#!/usr/bin/env python3
"""
Script to generate and save the KnowAI workflow Mermaid diagram.

Usage:
    python scripts/generate_workflow_diagram.py [output_file]

If no output file is specified, the diagram will be printed to stdout.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import knowai
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowai.core import get_workflow_mermaid_diagram


def main():
    """Generate and save the workflow diagram."""
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        print(f"Generating workflow diagram and saving to: {output_file}")
        diagram = get_workflow_mermaid_diagram(save_to_file=output_file)
        print(f"✅ Diagram saved to {output_file}")
    else:
        print("Generating workflow diagram...")
        diagram = get_workflow_mermaid_diagram()
        print("\n" + "="*50)
        print("KNOWAI WORKFLOW DIAGRAM")
        print("="*50)
        print(diagram)
        print("\n" + "="*50)
        print("To save this diagram to a file, run:")
        print("python scripts/generate_workflow_diagram.py output.md")


if __name__ == "__main__":
    main() 