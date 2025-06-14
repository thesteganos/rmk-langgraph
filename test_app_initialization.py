import sys
import os

# Ensure 'src' is in the Python path to allow importing from app
# This might be needed if 'app.py' does relative imports or expects a certain working directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Attempting to import app and initialize graph...")
try:
    # Import the function that sets up the graph, which was causing issues
    from app import get_graph # app.py should now have the asyncio fix

    # Attempt to call the function that initializes the graph
    # This is where the HuggingFace/torch components are initialized
    graph = get_graph()

    if graph:
        print("SUCCESS: app.get_graph() executed and returned a graph object.")
    else:
        print("WARNING: app.get_graph() executed but returned None or a falsy value.")

except RuntimeError as e:
    if "no running event loop" in str(e):
        print(f"FAILURE: Still encountered RuntimeError: {e}")
    else:
        print(f"Encountered an unexpected RuntimeError: {e}")
except ImportError as e:
    print(f"FAILURE: Could not import 'app' or 'get_graph': {e}")
    print("This might indicate an issue with the test script's path setup or a problem in app.py itself.")
except Exception as e:
    print(f"FAILURE: An unexpected error occurred: {type(e).__name__}: {e}")

print("Test script finished.")
