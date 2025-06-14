import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import sys

# Add src to path to allow direct import of modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # For ingest.py

from graph import WeightManagementGraph, GraphState # From src/graph.py
import ingest # The script itself

# Mocked LLM response for graph extraction
MOCK_LLM_GRAPH_EXTRACTION_RESPONSE_CONTENT = json.dumps({
    "entities": [
        {"name": "Metformin", "type": "Treatment"},
        {"name": "Type 2 Diabetes", "type": "Condition"}
    ],
    "relationships": [
        {"source": "Metformin", "target": "Type 2 Diabetes", "type": "TREATS"}
    ]
})

class AIMessageMock:
    def __init__(self, content):
        self.content = content

class TestNeo4jIntegration(unittest.TestCase):

    def setUp(self):
        # Environment variables for Neo4j and LLM
        self.patch_env = patch.dict(os.environ, {
            "NEO4J_URI": "bolt://mockneo4j:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
            "GOOGLE_API_KEY": "mock_api_key",
            "LLM_MODEL": "gemini-pro"
        })
        self.patch_env.start()

    def tearDown(self):
        self.patch_env.stop()

    @patch('ingest.ChatGoogleGenerativeAI')
    def test_extract_graph_data_from_chunk_success(self, MockChatGoogleGenerativeAI):
        mock_llm_instance = MockChatGoogleGenerativeAI.return_value
        mock_llm_instance.invoke.return_value = AIMessageMock(MOCK_LLM_GRAPH_EXTRACTION_RESPONSE_CONTENT)

        text_chunk = "Metformin is a treatment for Type 2 Diabetes."
        graph_data = ingest.extract_graph_data_from_chunk(text_chunk, mock_llm_instance)

        self.assertIsNotNone(graph_data)
        self.assertIn("entities", graph_data)
        self.assertIn("relationships", graph_data)
        self.assertEqual(len(graph_data["entities"]), 2)
        self.assertEqual(graph_data["entities"][0]["name"], "Metformin")
        self.assertEqual(len(graph_data["relationships"]), 1)
        self.assertEqual(graph_data["relationships"][0]["source"], "Metformin")

    @patch('ingest.ChatGoogleGenerativeAI')
    def test_extract_graph_data_from_chunk_json_error(self, MockChatGoogleGenerativeAI):
        mock_llm_instance = MockChatGoogleGenerativeAI.return_value
        mock_llm_instance.invoke.return_value = AIMessageMock("This is not JSON")

        text_chunk = "Some text."
        # Patch sys.stderr to capture print warnings
        with patch('sys.stderr', new_callable=unittest.mock.StringIO) as mock_stderr:
            graph_data = ingest.extract_graph_data_from_chunk(text_chunk, mock_llm_instance)
            self.assertIsNone(graph_data)
            self.assertIn("Error: Could not parse JSON from LLM response", mock_stderr.getvalue())

    @patch('ingest.GraphDatabase.driver')
    def test_store_graph_data_in_neo4j(self, MockNeo4jDriver):
        mock_driver_instance = MockNeo4jDriver.return_value
        mock_session = MagicMock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session # For 'with ... as session:'

        graph_data = json.loads(MOCK_LLM_GRAPH_EXTRACTION_RESPONSE_CONTENT)
        ingest.store_graph_data_in_neo4j(graph_data, mock_driver_instance)

        self.assertTrue(mock_session.run.called)
        # Check entity merge calls
        mock_session.run.assert_any_call(
            "MERGE (e:Entity {name: $name, type: $type})",
            name="Metformin", type="Treatment"
        )
        mock_session.run.assert_any_call(
            "MERGE (e:Entity {name: $name, type: $type})",
            name="Type 2 Diabetes", type="Condition"
        )
        # Check relationship merge call
        # Corrected the expected Cypher query string to match the implementation (using .format for rel_type)
        mock_session.run.assert_any_call(
            "MATCH (source:Entity {name: $source_name})\nMATCH (target:Entity {name: $target_name})\nMERGE (source)-[r:TREATS]->(target)",
            source_name="Metformin", target_name="Type 2 Diabetes"
        )

    @patch('src.graph.GraphDatabase.driver') # Patching where it's imported in src.graph
    def test_simple_neo4j_retriever_success(self, MockNeo4jDriverInGraph):
        mock_driver_instance = MockNeo4jDriverInGraph.return_value
        mock_driver_instance.verify_connectivity.return_value = None # Ensure connectivity check passes
        mock_session = MagicMock()
        mock_driver_instance.session.return_value.__enter__.return_value = mock_session

        mock_record1_data = {"entityName": "Ozempic", "entityType": "Drug", "relations_details": [{"name": "Weight Loss", "type": "Effect", "rel_type": "AIDS_IN", "start_node_name": "Ozempic", "end_node_name": "Weight Loss"}]}
        mock_record1 = MagicMock()
        mock_record1.get = lambda key, default=None: mock_record1_data.get(key, default)
        mock_record1.data.return_value = mock_record1_data # For direct data access if used

        mock_record2_data = {"entityName": "Obesity", "entityType": "Condition", "relations_details": []}
        mock_record2 = MagicMock()
        mock_record2.get = lambda key, default=None: mock_record2_data.get(key, default)
        mock_record2.data.return_value = mock_record2_data


        mock_session.run.return_value = [mock_record1, mock_record2]

        with patch.object(WeightManagementGraph, 'llm'), \
             patch.object(WeightManagementGraph, 'web_search_llm'), \
             patch.object(WeightManagementGraph, 'retriever'), \
             patch.object(WeightManagementGraph, 'llm_with_tools'):
            graph_manager = WeightManagementGraph()
            graph_manager.neo4j_driver = mock_driver_instance

            results = graph_manager.simple_neo4j_retriever("Ozempic")

        self.assertEqual(len(results), 2)
        self.assertIn("Entity: Ozempic (Drug). Related: Ozempic -[AIDS_IN]-> Weight Loss(Effect)", results[0])
        self.assertIn("Entity: Obesity (Condition). Related: No direct relations found matching the criteria.", results[1])
        mock_session.run.assert_called_once()


    @patch('src.graph.GraphDatabase.driver')
    def test_simple_neo4j_retriever_no_driver(self, MockNeo4jDriverInGraph):
        with patch.object(WeightManagementGraph, 'llm'), \
             patch.object(WeightManagementGraph, 'web_search_llm'), \
             patch.object(WeightManagementGraph, 'retriever'), \
             patch.object(WeightManagementGraph, 'llm_with_tools'):

            # Simulate Neo4j driver failing to initialize in __init__
            MockNeo4jDriverInGraph.driver.side_effect = Exception("Connection failed")
            # Or, more directly for this test, ensure it's None after __init__
            graph_manager = WeightManagementGraph()
            graph_manager.neo4j_driver = None

            results = graph_manager.simple_neo4j_retriever("query")
        self.assertEqual(results, [])

    @patch('src.graph.WeightManagementGraph.simple_neo4j_retriever')
    @patch('src.graph.Chroma') # Mock Chroma directly
    @patch('src.graph.ChatGoogleGenerativeAI')
    @patch('src.graph.StrOutputParser')
    @patch('src.graph.ChatPromptTemplate')
    def test_protocol_rag_node_with_neo4j(self, MockChatPromptTemplate, MockStrOutputParser, MockChatGoogleGenerativeAI, MockChroma, MockSimpleNeo4jRetriever):
        # Mock retrievers
        mock_chroma_instance = MockChroma.return_value.as_retriever.return_value # Mock the retriever object
        mock_chroma_doc = MagicMock()
        mock_chroma_doc.page_content = "Chroma document content."
        mock_chroma_instance.invoke.return_value = [mock_chroma_doc]

        MockSimpleNeo4jRetriever.return_value = ["Neo4j context string."]

        # Mock LLM and prompt chain
        mock_llm_instance = MockChatGoogleGenerativeAI.return_value
        mock_parser_instance = MockStrOutputParser.return_value
        mock_prompt_instance = MockChatPromptTemplate.from_template.return_value # from_template is called

        mock_chain_output = "Final LLM Answer"

        # Mock the runnable chain structure: prompt | llm | parser
        mock_prompt_llm_runnable = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_prompt_llm_runnable # prompt | llm
        mock_prompt_llm_runnable.__or__.return_value = mock_parser_instance # (prompt | llm) | parser
        mock_parser_instance.invoke.return_value = mock_chain_output # The end of the chain call

        with patch('src.graph.GraphDatabase.driver') as MockNeo4jDriverInGraphSetup:
            mock_driver_instance_setup = MockNeo4jDriverInGraphSetup.return_value
            mock_driver_instance_setup.verify_connectivity.return_value = None

            # We need to ensure that self.retriever is correctly mocked
            # The __init__ of WeightManagementGraph sets up self.retriever = Chroma(...).as_retriever(...)
            # So, we patch Chroma itself.
            with patch.dict(os.environ, {"DB_PATH": "dummy_db_path"}): # Ensure DB_PATH is set for Chroma init
                 graph_manager = WeightManagementGraph()

            # Replace instance methods/attributes with mocks AFTER instance creation
            graph_manager.retriever = mock_chroma_instance
            graph_manager.simple_neo4j_retriever = MockSimpleNeo4jRetriever
            graph_manager.llm = mock_llm_instance # Ensure the node uses the mocked LLM

            initial_state = GraphState(
                query="test query", user_profile={}, query_type="protocol",
                documents=[], web_results=[], final_answer="",
                disclaimer_needed=False, neo4j_results=[]
            )
            result_state = graph_manager.protocol_rag_node(initial_state)

        mock_chroma_instance.invoke.assert_called_with("test query")
        MockSimpleNeo4jRetriever.assert_called_with("test query")

        expected_combined_context = "Knowledge Base Documents (from vector search):\nChroma document content.\n\nKnowledge Graph Information (from graph database):\nNeo4j context string."
        # The invoke call is on the final part of the chain (parser_instance in this mock setup)
        mock_parser_instance.invoke.assert_called_with({"context": expected_combined_context, "question": "test query"})

        self.assertEqual(result_state["final_answer"], mock_chain_output)
        self.assertEqual(result_state["documents"], [mock_chroma_doc])
        self.assertEqual(result_state["neo4j_results"], ["Neo4j context string."])

if __name__ == '__main__':
    unittest.main()
