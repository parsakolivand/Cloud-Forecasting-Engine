"""Entry point for the app.

Run locally:   uv run streamlit run streamlit_app.py --server.address localhost
Hosted:        Streamlit Community Cloud runs this file directly from the repository.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cfe.app import main

main()
