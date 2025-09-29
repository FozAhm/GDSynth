# Import Libraries
import argparse
import enum
import logging
import random
import math
import json
import copy
import re
import time
import docker
from docker.models.containers import Container
from docker.errors import NotFound, APIError
from pathlib import Path
import nltk
from nltk.corpus import words
from neo4j import GraphDatabase
from datetime import datetime, timezone

class DatabaseType(enum.Enum):
    NEO4J = "neo4j"

class GraphTool(enum.Enum):
    REFINERY = "Refinery"

class TestTool(enum.Enum):
    GRAPHGENIE = "GraphGenie"

def parse_args():
    # Setup Parser
    parser = argparse.ArgumentParser(description='GraphDB Metamodel Extractor')

    parser.add_argument(
        "--database_type",
        type=str.lower,
        choices=[db.value for db in DatabaseType],
        help="The specific type of GraphDB from which you wish to extract the metamodel from",
        default=DatabaseType.NEO4J.value,
        required=False
    )

    parser.add_argument(
        "--database_ip",
        type=str,
        help='The IP of the GraphDB',
        required=True
    )

    parser.add_argument(
        "--database_port",
        type=str,
        help='The port of the GraphDB',
        required=True
    )

    parser.add_argument(
        "--database_username",
        type=str,
        help='The username to access the GraphDB',
        required=True
    )

    parser.add_argument(
        "--database_password",
        type=str,
        help='The password to access the GraphDB',
        required=True
    )

    parser.add_argument(
        "--log_level",
        type=str.lower,
        choices=["debug", "info"],
        help="The logging level you wish to use",
        default="debug",
        required=False
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="The seed you want to use in the random number generator used in graph mutations",
        default=3,
        required=False
    )

    parser.add_argument(
        "--search_space_file",
        type=str,
        help="The type of mutations you want to perform and the scale of the synthetic graphs",
        required=True
    )

    parser.add_argument(
    "--max_duration",
    type=int,
    help="The number of seconds to run a tests on graph models",
    default=3600
    )

    # Process and return the Arguements
    return parser.parse_args()

def configure_logging(name: str, log_level: str):
    # Create a new logger instance (not the root)
    logger = logging.getLogger(name)
    
    # Set the logging level based on input
    if log_level == "debug":
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)
    
    # Create and set formatter
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-5s %(message)s")
    
    # Create console handler with same level
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(formatter)
    
    # Clear existing handlers, then add ours
    logger.handlers.clear()
    logger.addHandler(ch)
    
    # Prevent logs bubbling up to root
    logger.propagate = False 
    
    return logger

class Metamodel:
    def __init__(self, node_labels, edge_labels, connectivity_matrix, all_edges, name="original", target_size=30, random_seed=3, graph_tool=GraphTool.REFINERY.value, test_tool=TestTool.GRAPHGENIE.value, graph_db=DatabaseType.NEO4J.value, versions=[]):
        # Setup Graph Data Structures
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.connectivity_matrix = connectivity_matrix
        self.all_edges = all_edges

        # Set the metamodel name
        self.name = name

        # Set the target size of the graph should the metamodel need to be generated
        self.target_size = target_size

        # Set the Graph Generation Tool
        self.graph_tool = GraphTool(graph_tool)

        # Set the location of the input/output file for the graph generation tool
        self.graph_tool_input_file = None
        self.graph_tool_output_file = None

        # Set the Graph DB Testing Tool
        self.test_tool = TestTool(test_tool)

        # Set the GDBMS under Test and its versions
        self.graph_db = DatabaseType(graph_db)
        self.versions = versions

        # Set the random seed for the local random generator for reproducibility
        self.random_generator = random.Random(random_seed)

        # Download required NLTK data for random node generation
        nltk.download('words', quiet=True)
        # Get the list of all English words from the NLTK corpus
        self.word_list = words.words()
    
    # Define setters
    def set_name(self, name: str):
        self.name = name
    
    def set_target_size(self, target_size: int):
        self.target_size = target_size
    
    def set_graph_tool(self, graph_tool):
        self.graph_tool = GraphTool(graph_tool)
    
    def set_graph_tool_input_file(self, graph_tool_input_file: Path):
        self.graph_tool_input_file = graph_tool_input_file
    
    def set_graph_tool_output_file(self, graph_tool_output_file: Path):
        self.graph_tool_output_file = graph_tool_output_file
    
    def set_test_tool(self, test_tool):
        self.test_tool = TestTool(test_tool)
    
    def set_graph_db(self, graph_db):
        self.graph_db = DatabaseType(graph_db)
    
    def set_versions(self, versions):
        self.versions = versions
    
    # Do nothing mutation
    def original(self):
        return None
    
    # Adds an existing edge type between two ramdom nodes
    def add_edge(self):
        # Choose a random edge label
        random_edge_label = self.random_generator.choice(self.edge_labels)

        # Select a random source and target node label from the existing labels
        source_node_label = self.random_generator.choice(self.node_labels)
        target_node_label = self.random_generator.choice(self.node_labels)

        self.connectivity_matrix[source_node_label][target_node_label].add(random_edge_label)

        return source_node_label, random_edge_label, target_node_label
    
    # Adds an edge in reverse of an existing edge
    def add_edge_reversed(self):
        # If no edges exist, return
        if not self.all_edges:
            print("No edges to add in reverse.")
            return None

        # Randomly select one edge
        random_edge = self.random_generator.choice(self.all_edges)
        source_node_label, target_node_label, edge_label = random_edge

        # Add Edge in reverse
        self.connectivity_matrix[target_node_label][source_node_label].add(edge_label)

        return target_node_label, edge_label, source_node_label
    
    # Adds an edge type and creates an edge between two random node types
    def add_edge_label(self):
        # Create a new edge type label
        new_edge_label = self.generate_new_word(self.edge_labels)
        self.edge_labels.append(new_edge_label)

        # Select a random source and target node label from the existing labels
        source_node_label = self.random_generator.choice(self.node_labels)
        target_node_label = self.random_generator.choice(self.node_labels)

        self.connectivity_matrix[source_node_label][target_node_label].add(new_edge_label)

        return source_node_label, new_edge_label, target_node_label

    # Creates a new node type and creates an edge to a random existing target node type
    def add_node_label(self):
        # Generate a new node label
        new_node_label = self.generate_new_word(self.node_labels)

        # Add this new label to the node_labels list
        self.node_labels.append(new_node_label)

        # Add the new label to the connectivity matrix
        self.connectivity_matrix[new_node_label] = {end_label: set() for end_label in self.node_labels}

        for node_label in self.node_labels:
            self.connectivity_matrix[node_label][new_node_label] = set()

        # Create a new edge type label
        new_edge_label = self.generate_new_word(self.edge_labels)
        self.edge_labels.append(new_edge_label)

        # Select a random target node label from the existing labels
        target_node_label = self.random_generator.choice(self.node_labels)

        # Add the new edge type between the the new node type and the randomly selected target node type
        self.connectivity_matrix[new_node_label][target_node_label].add(new_edge_label)

        return new_node_label, new_edge_label, target_node_label
    
    # Flip the direction of an edge an random
    def change_edge_direction(self):
        # If no edges exist, return
        if not self.all_edges:
            print("No edges to reverse")
            return None
        
        # Randomly select one edge
        random_edge = self.random_generator.choice(self.all_edges)
        source_node_label, target_node_label, flip_edge_label = random_edge

        # Remove edge from existing node pair
        self.connectivity_matrix[source_node_label][target_node_label].discard(flip_edge_label)

        # Add edge to reversed node pair
        self.connectivity_matrix[target_node_label][source_node_label].add(flip_edge_label)

        return target_node_label, flip_edge_label, source_node_label
    
    # Deletes an edge type and all edges of that type
    def delete_edge_label(self):
        # Choose a random edge label
        delete_edge_label = self.random_generator.choice(self.edge_labels)

        # Remove the edge label from the list
        self.edge_labels.remove(delete_edge_label)

        # Delete edge label from connectivity matrix
        for source_node_label in self.node_labels:
            for target_node_label in self.connectivity_matrix[source_node_label]:
                self.connectivity_matrix[source_node_label][target_node_label].discard(delete_edge_label)
        
        return delete_edge_label
    
    # Deletes a node type and all associated edges
    def delete_node_label(self):
        # Select a random node label from the existing labels to delete
        delete_node_label = self.random_generator.choice(self.node_labels)

        # Remove the node label from the list
        self.node_labels.remove(delete_node_label)
        
        # Remove the node label as a target for all relationships
        for node_label in self.node_labels:
            del self.connectivity_matrix[node_label][delete_node_label]
        
        # Remove node label as a source for all relationships
        del self.connectivity_matrix[delete_node_label]

        # Recompute the edge label set
        temp_edge_labels = set()

        for source_node_label in self.node_labels:
            for target_node_label in self.connectivity_matrix[source_node_label]:
                for edge_type in self.connectivity_matrix[source_node_label][target_node_label]:
                    temp_edge_labels.add(edge_type)
        
        self.edge_labels = list(temp_edge_labels)

        return delete_node_label
    
    # Moves an edge from one source/target nodes to another pair
    def move_edge(self):
        # If no edges exist, return
        if not self.all_edges:
            print("No edges to move.")
            return None

        # Randomly select one edge
        random_edge = self.random_generator.choice(self.all_edges)
        source_node_label, target_node_label, move_edge_label = random_edge

        # Remove edge from existing node pair
        self.connectivity_matrix[source_node_label][target_node_label].discard(move_edge_label)

        # Select a random source and target node label from the existing labels
        random_source_node_label = random.choice(self.node_labels)
        random_target_node_label = random.choice(self.node_labels)

        # Add edge to a new node pair
        self.connectivity_matrix[random_source_node_label][random_target_node_label].add(move_edge_label)

        return random_source_node_label, move_edge_label, random_target_node_label
        
    def generate_new_word(self, existing_words):
        # Convert the existing words to a set for faster lookup
        existing_words_set = set(existing_words)
        
        # Retry mechanism to find a new word
        attempts = 0
        new_word = None
        while not new_word and attempts < 100:
            # Select a random word from the word list
            random_word = self.random_generator.choice(self.word_list).lower()
            
            # Ensure the new word is not in the existing words
            if random_word not in existing_words_set:
                new_word = random_word
            
            attempts += 1
        
        return new_word

class MetamodelExtractor:
    def __init__(self, args, logger: logging.Logger):
        # Database Properties
        self.logger = logger
        self.ip = args.database_ip
        self.logger.debug("Database IP: " + self.ip)
        self.port = args.database_port
        self.logger.debug("Database Port: " + self.port)
        self.username = args.database_username
        self.logger.debug("Database Username: " + self.username)
        self.password = args.database_password
        self.logger.debug("Database Password: " + self.password)
        self.random_seed = args.random_seed
        self.logger.debug("Metamodel Random Generator Seed: " + str(self.random_seed))

        self.database_type = DatabaseType(args.database_type)
        self.logger.info("Extracting data from the following database: " + str(self.database_type.value))

        # Setup Graph Data Structures
        self.node_labels = []
        self.edge_labels = []
        self.connectivity_matrix = {}
        self.all_edges = []
    
    def extract(self):
        if self.database_type == DatabaseType.NEO4J:
            self.logger.debug("Running Neo4j extraction")
            self.neo4j()
        # To support more DatabaseType values later, handle them here
        else:
            self.logger.error("Unsupported database type: " + str(self.database_type.value))
            raise ValueError(f"Unsupported database type: {self.database_type.value}")

        # Gather all the edges
        self.get_all_edges()

        self.logger.info("Extraction complete")
        self.logger.info(self.node_labels)
        self.logger.info(self.edge_labels)
        self.logger.info(self.connectivity_matrix)

        return Metamodel(self.node_labels, self.edge_labels, self.connectivity_matrix, self.all_edges, random_seed=self.random_seed)
    
    def get_all_edges(self):
        # Collect all edges in the format (source_node_label, target_node_label, edge_label)
        for source_node_label in self.node_labels:
            for target_node_label in self.connectivity_matrix[source_node_label]:
                for edge_label in self.connectivity_matrix[source_node_label][target_node_label]:
                    self.all_edges.append((source_node_label, target_node_label, edge_label))
    
    def neo4j(self):
        # Setup Datebase Connection
        url = "bolt://{}:{}".format(self.ip, self.port)
        driver = GraphDatabase.driver(url, auth=(self.username, self.password))

        # Setup the Queries
        get_node_labels_query = """
            CALL db.labels() YIELD label
            RETURN label
            ORDER BY label;
        """

        get_edge_labels_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN relationshipType
            ORDER BY relationshipType;
        """

        # Get Connectivity Matrix
        get_connectivity_query = """
            MATCH (a)-[r]->(b)
            RETURN DISTINCT labels(a) AS startLabel, type(r) AS relationshipType, labels(b) AS endLabel
        """

        with driver.session() as session:
            # Get Node Labels 
            node_result = session.run(get_node_labels_query)
            for record in node_result:
                self.logger.debug(record)
                self.node_labels.append(record['label'])
            
            # Get the Edge Labels
            edge_result = session.run(get_edge_labels_query)
            for record in edge_result:
                self.logger.debug(record)
                self.edge_labels.append(record['relationshipType'])

            # Initialize the connectivity matrix
            for start_label in self.node_labels:
                self.connectivity_matrix[start_label] = {end_label: set() for end_label in self.node_labels}

            # Get Connectivity Data
            connectivity_result = session.run(get_connectivity_query)
            for record in connectivity_result:
                self.logger.debug(record)
                start_labels = record['startLabel']
                relationship_type = record['relationshipType']
                end_labels = record['endLabel']
                for start_label in start_labels:
                    for end_label in end_labels:
                        self.connectivity_matrix[start_label][end_label].add(relationship_type)

class MetamodelMutator:
    def __init__(self, args, metamodel: Metamodel, logger: logging.Logger):
        #Set the Logger
        self.logger = logger

        # Store original metamodel for cloning
        self.original_metamodel = metamodel

        # Setup an array to store mutated metamodels
        self.mutated_metamodels = []

        # Store where the search space definition file is
        self.search_space_file = args.search_space_file
        self.logger.info("The search space definition file location: " + self.search_space_file)

        # Load the search space definition file
        with open(self.search_space_file, 'r') as f:
            self.search_space = json.load(f)
        
        # Map JSON mutation names to Metamodel methods
        self._mutation_map = {
            "original": "original",
            "add_edge": "add_edge",
            "add_edge_reversed": "add_edge_reversed",
            "add_edge_type": "add_edge_label",
            "add_node_type": "add_node_label",
            "change_edge_direction": "change_edge_direction",
            "delete_edge_type": "delete_edge_label",
            "delete_node_type": "delete_node_label",
            "move_edge": "move_edge"
        }

        # Valid lists for tools and databases
        self.valid_graph_tools = [tool.value for tool in GraphTool]
        self.valid_test_tools = [tool.value for tool in TestTool]
        self.valid_gdbms = [db.value for db in DatabaseType]
    
    def mutate(self):
        mutation_candidate_counter = 1
        # Iterate over the mutation candidates
        for mutation_candidate in self.search_space["DesiredMutationCandidates"]:
            mutation_candidate_name = mutation_candidate["Name"]
            self.logger.info(f"Working on Root Mutation Candidate #{mutation_candidate_counter}: {mutation_candidate_name}")
            
            self.logger.info(f"Requested Target Sizes for Root Mutation Candidate #{mutation_candidate_counter}: {mutation_candidate['SizeScope']}")

            # Validate GraphTool
            gt = mutation_candidate["GraphTool"]
            if gt not in self.valid_graph_tools:
                self.logger.error(f"Unknown GraphTool '{gt}' in search-space file at root mutation candidate '{mutation_candidate_name}'")
                raise ValueError(f"Unknown GraphTool '{gt}' in search-space file at root mutation candidate '{mutation_candidate_name}'")
            # Validate TestTool
            tt = mutation_candidate["TestTool"]
            if tt not in self.valid_test_tools:
                self.logger.error(f"Unknown TestTool '{tt}' in search-space file at root mutation candidate '{mutation_candidate_name}'")
                raise ValueError(f"Unknown TestTool '{tt}' in search-space file at root mutation candidate '{mutation_candidate_name}'")
            # Validate GDBMS
            gdb = mutation_candidate["GDBMS"]
            if gdb not in self.valid_gdbms:
                self.logger.error(f"Unknown GDBMS '{gdb}' in search-space file at root mutation candidate '{mutation_candidate_name}'")
                raise ValueError(f"Unknown GDBMS '{gdb}' in search-space file at root mutation candidate '{mutation_candidate_name}'")


            sub_mutation_candidate_counter = 1
            # Iterate over the size of graphs wanted from each metamodel candidate
            for mutation_candidate_size in mutation_candidate["SizeScope"]:
                # Create a Metamodel based on the original for the mutation candidate
                mutation_candidate_metamodel = copy.deepcopy(self.original_metamodel)

                # Set the size
                mutation_candidate_metamodel.set_target_size(int(mutation_candidate_size))
                self.logger.info(f"Working on Sub Mutation Candidate for Root Mutation Candidate #{mutation_candidate_counter} with target size: {mutation_candidate_metamodel.target_size}")

                # Set the name
                sub_mutation_candidate_name = mutation_candidate_name + "_" + str(mutation_candidate_size)
                mutation_candidate_metamodel.set_name(sub_mutation_candidate_name)
                self.logger.info(f"Sub Mutation Candidate Name: {mutation_candidate_metamodel.name}")

                # Set Graph Generation Tool
                mutation_candidate_metamodel.set_graph_tool(gt)

                # Set the Graph Testing Tool
                mutation_candidate_metamodel.set_test_tool(tt)

                # Set the GDBMS under test
                mutation_candidate_metamodel.set_graph_db(gdb)

                # Set the GDBMS versions under test
                mutation_candidate_metamodel.set_versions(mutation_candidate["Versions"])

                # Loop through the mutation operations requested for this candidate
                for mutation_operation in mutation_candidate["MutationArray"]:
                    self.logger.debug(f"Performing mutation operation: {mutation_operation}")
                    # Get the mapping of the mutation operations requested and the internal names of the mutation functions of a metamodel
                    mutation_operation_method_name = self._mutation_map.get(mutation_operation)

                    # Raise exception if error in search space file
                    if not mutation_operation_method_name:
                        self.logger.error(f"Unknown mutation '{mutation_operation_method_name}' in search-space file")
                        raise ValueError(f"Unknown mutation '{mutation_operation_method_name}' in search-space file")
                    
                    # Get the mutatation method
                    mutation_operation_method = getattr(mutation_candidate_metamodel, mutation_operation_method_name, None)
                    # Raise exception if the mutation method is not available for Metamodels
                    if not mutation_operation_method:
                        self.logger.error(f"Metamodel has no method '{mutation_operation_method}'")
                        raise AttributeError(f"Metamodel has no method '{mutation_operation_method}'")
                    
                    # Excute the mutation
                    mutation_operation_method()
                
                # Append the fully mutated mutation candidate
                self.mutated_metamodels.append(mutation_candidate_metamodel)
                self.logger.info(f"Finished Sub Mutation Candidate {mutation_candidate_metamodel.name}")
                sub_mutation_candidate_counter += 1

            
            self.logger.info(f"Finished mutating Root Mutation Candidate #{mutation_candidate_counter}: {mutation_candidate_name}")
            mutation_candidate_counter += 1
    
        return self.mutated_metamodels
    
class MetamodelAdapter:
    def __init__(self, args, metamodels: list[Metamodel], logger: logging.Logger):
        #Set the Logger
        self.logger = logger

        # Setup the random seed
        self.random_seed = args.random_seed
        self.logger.debug(f"Metamodel Random Generator Seed: {self.random_seed}")
        
        # Load the Metamodels
        self.metamodels = metamodels

        # Setup an array to store adapted metamodels
        self.adapted_metamodels = []

        # Get the current working directory
        self.cwd = Path.cwd()
        self.logger.info(f"Current working directory: {self.cwd}")

        # Setup folders for the different types of graph generation tools
        self.input_folder_name = "Input"
        self.output_folder_name = "Output"
        for graph_tool in GraphTool:
            # List out the directories you want to ensure exist
            paths_to_create = [
                Path(graph_tool.value) / self.input_folder_name,
                Path(graph_tool.value) / self.output_folder_name,
            ]
            for path in paths_to_create:
                path.mkdir(parents=True, exist_ok=True)
        
        self.docker_client = docker.from_env()

        self._docker_image_map = {
            GraphTool.REFINERY: "ghcr.io/graphs4value/refinery-cli:latest",
        }
    
    def adapt(self):
        adapted_metamodel_counter = 1
        for metamodel in self.metamodels:
            metamodel_name = metamodel.name
            self.logger.info(f"Working to adapt metamodel #{adapted_metamodel_counter}: {metamodel_name}")
            # Get the graph generation tool
            metamodel_graph_tool = metamodel.graph_tool
            self.logger.info(f"Graph Generation Tool Selected: {metamodel_graph_tool.value}")

            if metamodel_graph_tool == GraphTool.REFINERY:
                # Convert the metamodel to Refinery problem
                refinery_problem = self.convert_to_refinery(metamodel)
                
                # Save Refinery problem to file as the tool only takes file inputs
                # Set the path of the refinery problem
                refinery_problem_file_path = Path(metamodel_graph_tool.value) / self.input_folder_name / f"{metamodel_name}.problem"
                # Update metamodel with the refinery problem path
                metamodel.set_graph_tool_input_file(refinery_problem_file_path)
                # Save the file
                refinery_problem_location = self.save_to_file(refinery_problem, refinery_problem_file_path)
                self.logger.info(f"Metamodel saved as Refinery problem at: {refinery_problem_location.resolve()}")

                # Generate refinery solution 
                # Set the path of the refinery problem
                refinery_solution_file_path = Path(metamodel_graph_tool.value) / self.output_folder_name / f"{metamodel_name}_r{self.random_seed}.solution"
                # Update metamodel with the refinery solution path
                metamodel.set_graph_tool_output_file(refinery_solution_file_path)
                # Get the Refinery solution
                refinery_solution_location = self.generate_with_refinery(metamodel)
                self.logger.info(f"Graph for metamodel #{adapted_metamodel_counter} {metamodel_name} has finished generating")
                self.logger.info(f"Graph model saved as Refinery solution at: {refinery_solution_location.resolve()}")
            
            self.adapted_metamodels.append(metamodel)
            adapted_metamodel_counter += 1
        
        self.logger.info(f"{len(self.adapted_metamodels)} Metamodel Candidates converted into graph models")

        return self.adapted_metamodels

    # A file writer that takes a list and writes it line by line, we assume its a list of strings
    def save_to_file(self, contents: list[str], location: Path):
        # If the file already exists, delete it
        if location.exists():
            location.unlink()
        
        # Make sure the parent directory is there
        location.parent.mkdir(parents=True, exist_ok=True)

        with location.open('w') as file:
            for line in contents:
                file.write(f"{line}\n")
        
        return location

    def generate_with_refinery(self, metamodel: Metamodel):
        # Get the refinery problem path
        refinery_problem_location = metamodel.graph_tool_input_file
        # identify the Refinery folder as the base
        refinery_base_dir = refinery_problem_location.parents[1]
        refinery_problem_rel_path = refinery_problem_location.relative_to(refinery_base_dir)

        refinery_solution_location = metamodel.graph_tool_output_file
        refinery_solution_rel_path = refinery_solution_location.relative_to(refinery_base_dir)

        docker_image_name = self._docker_image_map.get(metamodel.graph_tool)
        self.logger.debug(f"Using {docker_image_name} docker image for refinery")

        mapping_dir = str(refinery_solution_location.parent.parent.resolve())
        self.logger.debug(f"Refinery will be mapped to the follow local directory: {mapping_dir}")

        input_command = f"generate {refinery_problem_rel_path} -o {refinery_solution_rel_path} -r {self.random_seed}"
        self.logger.debug(f"Refinery Docker Input Command: {input_command}")

        self.docker_client.containers.run(
            image=docker_image_name,
            command=input_command,
            auto_remove=True,
            # detach=True,
            volumes={
                mapping_dir: {
                    "bind": "/data",
                    'mode': 'rw'
                }
            }
        )

        return refinery_solution_location

    def convert_to_refinery(self, metamodel: Metamodel):
        # Load Graph Data Structures
        node_labels = metamodel.node_labels
        connectivity_matrix = metamodel.connectivity_matrix

        # Set the default size of graph to generate
        scope=metamodel.target_size
    
        # Setup empty list for lines of reinfery problem
        refinery_problem = []

        # Get the number of node labels
        num_node_labels = len(node_labels)
        # Based on the scope of the graph and mumber of node labels set the minimum required number of nodes of each type
        min_node_scope = int(scope/2/num_node_labels)
        # Based on the minimum number of nodes of each type, set the maxium number of edges for each type for each node
        max_edge_scope = int(math.sqrt(min_node_scope))
        if max_edge_scope < 10:
            max_edge_scope = 10

        # Number each edge (even if the same) as refinery deals wtih each edge being different
        edge_counter = 0
        for node_label in node_labels:
            # define a temp defintion of the node label in refinery format
            node_refinery_definition = []
            # count the number of outgoing edges for this node label 
            node_edge_counter = 0

            self.logger.debug("For Source Node: " + node_label)

            for target_node_label in connectivity_matrix[node_label]:
                edges = connectivity_matrix[node_label][target_node_label]
                
                self.logger.debug("To Target Node: " + target_node_label)
                
                if len(edges) == 0:
                    self.logger.debug("No Edges Exist")
                else:
                    for edge in edges:
                        node_edge_counter += 1
                        edge_counter += 1
                        self.logger.debug("Edge #" + str(node_edge_counter) + ": " + str(edge))
                        node_refinery_definition.append('    ' + str(target_node_label) + '[1..' + str(max_edge_scope) + '] ' + str(edge) + str(edge_counter))
            
            if node_edge_counter == 0:
                node_refinery_definition.insert(0, 'class ' + str(node_label) + '.')
            else:
                node_refinery_definition.insert(0, 'class ' + str(node_label) + '{')
                node_refinery_definition.append('}')
            
            # Add a newline
            node_refinery_definition.append('')
            # Append node defintion to refinery problem data deifntion
            refinery_problem = refinery_problem + node_refinery_definition
        
        # Add a global node scope
        refinery_scope = "scope node = " + str(scope) + ".." + str(int(scope*1.1))
        # Add a scope for each node type
        for node_label in node_labels:
            refinery_scope = refinery_scope + ", " + node_label + " = " + str(min_node_scope) + "..*"
        # Complete Scope String
        refinery_scope = refinery_scope + "."
        # Add scope statement to file
        refinery_problem.append(refinery_scope)

        refinery_problem.append('')

        # for line in refinery_problem:
        #     logger.debug(line)
        
        return refinery_problem

class TestCase():
    def __init__(self, name: str, graph_db: GraphDatabase, version: str, test_tool: TestTool):
        # Set the name of the test
        self.name = name

        # Set the GraphDB Type under test
        self.graph_db = graph_db

        # Set the GraphDB Version under test
        self.version = version

        # Set the Test Tool being used
        self.test_tool = test_tool
     
    def set_db_container(self, db_container: Container):
        self.db_container = db_container
    
    def set_test_container(self, test_container: Container):
        self.test_container = test_container
    
    def set_start_time(self, start_time: datetime):
        self.start_time = start_time

class TestCoordinator():
    def __init__(self, args, graphmodels: list[Metamodel], logger: logging.Logger):
        # Set the Logger
        self.logger = logger

        # Load the Adapted Metamodels
        self.graphmodels = graphmodels

        # Get the Docker Client
        self.docker_client = docker.from_env()

        # Setup the Network
        # The network name to use
        self.docker_network_name = "graph_test_network"
        # See if it already exists
        docker_network_list = self.docker_client.networks.list(names=[self.docker_network_name])
        # Create a new network if it does not otherwise use existing
        if len(docker_network_list) == 0:
            self.logger.debug("Creating a new Docker network")
            self.docker_network = self.docker_client.networks.create(self.docker_network_name, driver="bridge", check_duplicate=True) 
        else:
            self.logger.debug("Using existing Docker network")
            self.docker_network = docker_network_list[0]
        
        self.logger.info(f"Using Docker network {self.docker_network_name} with ID {self.docker_network.short_id}")

        # Create an Array of Tests
        self.test_cases: list[TestCase] = []

        # Set the maximum run time & update frequency
        self.max_duration = args.max_duration
        self.update_frequency = 300

        # Set the GDBMS Container Images Map
        self._graphdb_image_map = {
            DatabaseType.NEO4J: {
                "5.1.0": "neo4j:5.1.0-community",
                "5.4.0": "neo4j:5.4.0-community"
            }
        }

        # Set the Test Tool Container Images Map
        self._test_tool_image_map = {
            TestTool.GRAPHGENIE: "fozailahmadmcgill/gdsynth:GraphGenie"
        }

        # Create Results Directory
        self.results_folder = Path("Results")
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Set the result files to retrieve for each Test Tool
        self._test_tool_file_map = {
            TestTool.GRAPHGENIE: [
                "/code/GraphGenie/testing.log",
                "/code/GraphGenie/bug.log",
                "/code/GraphGenie/exception.log"
            ]
        }
        
        # Keep an incrementing value for the GDBMS port to avoid conflicts
        self.gdbms_port_counter = 1
    
    def start_campaign(self):
        for graph_model in self.graphmodels:
            self.logger.info(f"Starting Test on Graph Model: {graph_model.name}")
            
            # Graph Database under test 
            test_graph_db_type = graph_model.graph_db

            # Graph Test Tool being used
            test_graph_tool = graph_model.test_tool 

            # Start a test for each version of the GDBMS requested 
            for test_graph_db_version in graph_model.versions:
                # Create a Test Case
                test_name = str(graph_model.graph_tool_output_file.stem + "_" + test_graph_db_type.value + "_" + test_graph_db_version)
                test_case = TestCase(test_name, test_graph_db_type, test_graph_db_version, test_graph_tool)

                # Start Test based on the type of GDBMS under test
                if test_graph_db_type == DatabaseType.NEO4J:
                    self.logger.info(f"Testing GDBMS: {test_graph_db_type.value} version {test_graph_db_version}")
                    
                    # Determine the docker image name to use
                    graphdb_docker_image_name = self._graphdb_image_map.get(test_graph_db_type).get(test_graph_db_version)
                    
                    if not graphdb_docker_image_name:
                        self.logger.error(f"No Docker Image availble for GDBMS {test_graph_db_type} version {test_graph_db_version}")
                        raise ValueError(f"No Docker Image availble for GDBMS {test_graph_db_type} version {test_graph_db_version}")

                    self.logger.debug(f"Using Docker Image: {graphdb_docker_image_name}")

                    # Set the name for the GDBMS under test container
                    graphdb_docker_container_name = str(graph_model.graph_tool_output_file.stem + "_" + test_graph_db_type.value + "_" + test_graph_db_version)

                    username = "neo4j"
                    self.logger.debug(f"Username: {username}")
                    password = "mcgill123!"
                    self.logger.debug(f"Password: {password}")
                    http_port = 7474
                    http_port_host = int(http_port+self.gdbms_port_counter)
                    bolt_port = 7687
                    bolt_port_host = int(bolt_port+self.gdbms_port_counter)

                    # Start the GDBMS under test
                    gdbms_container = self.docker_client.containers.run(
                        image=graphdb_docker_image_name,
                        environment={
                            "NEO4J_AUTH": f"{username}/{password}"
                        },
                        name=graphdb_docker_container_name,
                        hostname=graphdb_docker_container_name,
                        detach=True,
                        ports={
                            http_port: http_port_host,
                            bolt_port: bolt_port_host
                        },
                        network=self.docker_network.name
                    )
                    # Give Time for the GDBMS container to startup
                    time.sleep(30)
                    gdbms_container = self.docker_client.containers.get(gdbms_container.id)

                    # Set the GDBMS container for the test case
                    self.logger.info(f"Started GDBMS Container: {graphdb_docker_container_name}")
                    test_case.set_db_container(gdbms_container)

                    # Get the IP of the GDBMS container
                    gdbms_container_ip = gdbms_container.attrs["NetworkSettings"]["Networks"][self.docker_network_name]["IPAddress"]
                    self.logger.debug(f"GDBMS Container IP: {gdbms_container_ip}")

                    # Load the GDBMS with the synthetic data
                    self.load_graph_model(graph_model, bolt_port_host, username, password)

                    # Setup and start the test tool
                    if test_graph_tool == TestTool.GRAPHGENIE:
                        self.logger.info(f"Using GDBMS Test Tool: {test_graph_tool.value}")
                        # Determine the docker image name to use for the test tool
                        test_tool_docker_image_name = self._test_tool_image_map.get(test_graph_tool)
                        
                        if not test_tool_docker_image_name:
                            self.logger.error(f"No Docker Image availble for Test Tool {test_tool_docker_image_name}")
                            raise ValueError(f"No Docker Image availble for Test Tool {test_tool_docker_image_name}")

                        # Set the name for the GDBMS under test container
                        test_tool_docker_container_name = str(graph_model.graph_tool_output_file.stem + "_" + test_graph_tool.value)

                        test_tool_command = "--graph-db-ip "+ gdbms_container_ip + " --graph-db-username " + username + " --graph-db-password " + password
                        self.logger.debug(f"Test Tool Command: {test_tool_command}")

                        test_container = self.docker_client.containers.run(
                            image=test_tool_docker_image_name,
                            name=test_tool_docker_container_name,
                            hostname=test_tool_docker_container_name,
                            detach=True,
                            network=self.docker_network.name,
                            command=f"--graph-db-ip {gdbms_container_ip} --graph-db-username {username} --graph-db-password {password}"
                        )

                        # Give Time for the Test Tool container to startup
                        time.sleep(10)
                        test_container = self.docker_client.containers.get(test_container.id)

                        # Set the test container
                        self.logger.info(f"Started Test Container: {test_tool_docker_container_name}")
                        test_case.set_test_container(test_container)

                        # Set the start time
                        test_case.set_start_time(datetime.now(timezone.utc))

                        # Appent the test case to the list of test cases running
                        self.test_cases.append(test_case)
                    else:
                        self.logger.error(f"We have not implemented Test Tool: {test_graph_tool}")
                        break
                else:
                    self.logger.error(f"We have not implemented GDBMS: {test_graph_db_type}")
                    break
                
                # Increment the GDBMS Port Counter
                self.gdbms_port_counter += 1
    
    def wait(self):
        self.logger.info(f"Test Campaign will run for: {self.max_duration} seconds")

        time_counter = 0

        num_wait_blocks = int(self.max_duration / self.update_frequency)
        remaining_time = int(self.max_duration % self.update_frequency)

        for i in range(num_wait_blocks):
            time.sleep(self.update_frequency)
            time_counter += self.update_frequency
            self.logger.info(f"Test Campaign has been running for: {time_counter} seconds")

        time.sleep(remaining_time)
        time_counter += remaining_time

        return time_counter
    
    def get_results(self):
        for test in self.test_cases:
            self.logger.info(f"Getting results for Test: {test.name}")

            # Create Destination Folder
            test_result_folder = self.results_folder / test.name
            test_result_folder.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created test results folder: {test_result_folder}")

            # Stop the test container
            self.logger.info(f"Stopping Test Container: {test.test_container.name}")
            test.test_container.stop()

            # Stop Remove the GDBMS Container
            self.logger.info(f"Stopping GDBMS Container: {test.db_container.name}")
            test.db_container.stop()
            test.db_container.remove()

            # Get all the result files from the test container
            if test.test_tool == TestTool.GRAPHGENIE:
                for result_file in self._test_tool_file_map.get(test.test_tool):
                    self.logger.debug(f"Getting test result file from container: {result_file}")
                    self.get_file(test.test_container, result_file, test_result_folder)
            else:
                self.logger.error(f"We have not implemented Test Tool: {test.test_tool}")
            
            # Remove the Test Container
            test.test_container.remove()

            self.logger.info("All test containers have been removed")
        
        self.logger.info("All test results retrieved")
    
    def cleanup(self):
        self.logger.debug("Removing Docker Network for test environments")
        self.docker_network.remove()

    
    def load_graph_model(self, graphmodel: Metamodel, port: int, username: str, password: str, ip="localhost"):
        # Data Loading Routine for Neo4j and Refinery
        if graphmodel.graph_db == DatabaseType.NEO4J and graphmodel.graph_tool == GraphTool.REFINERY:
            url = "bolt://{}:{}".format(ip, port)
            self.logger.debug(f"Connecting to DB: {url}")
            driver = GraphDatabase.driver(url, auth=(username, password))

            solution_file_path = graphmodel.graph_tool_output_file.resolve()
            self.logger.info(f"Using solution file: {solution_file_path}")
            solution_file = open(solution_file_path, 'r')
            solution_file_lines = solution_file.readlines()
            solution_file_lines_stripped = []

            start_index = 0
            count = 0
            # Strips the newline character
            for line in solution_file_lines:
                line_stripped = line.strip()
                solution_file_lines_stripped.append(line_stripped)
                if 'declare' in line_stripped:
                    self.logger.debug(f"Found start of graph data: {count}")
                    start_index = count
                
                count += 1
            
            graph_instance = solution_file_lines_stripped[start_index:]

            # node_regex = "([a-zA-Z])\w+\(([a-zA-Z])\w+\)."
            node_regex = "^(\w+)\((\w+)\)\.$"
            # transition_regex = "([a-zA-Z])\w+\(([a-zA-Z])\w+, ([a-zA-Z])\w+\)."
            transition_regex = "^(\w+)\((\w+),\s*(\w+)\)\.$"
            delimiters = ["(", ")", ",", "."]

            node_types = set()
            num_nodes = 0
            transition_types = set()
            num_edges = 0
            for line in graph_instance:

                if re.search(node_regex, line):
                    regex_match = re.search(node_regex, line)
                    node_type = regex_match.group(1)  # This will be 'Actor'
                    node_id = regex_match.group(2)  # This will be 'actor5'
                    # print("Node Line: ", line)
                    self.logger.debug(f"Node Type: {node_type} | Node ID: {node_id}")
                    node_types.add(node_type)
                    num_nodes += 1
                    with driver.session() as session:
                        create_node_query = "MERGE ({}:{} {{rid: '{}'}})".format(node_id, node_type, node_id)
                        self.logger.debug(f"Create Node Query: {create_node_query}")
                        session.run(create_node_query)
                elif re.search(transition_regex, line):
                    regex_match = re.search(transition_regex, line)
                    # Get the transition type
                    transition_type = regex_match.group(1)
                    # Remove the Refinery specific numbering for transitions
                    transition_type = transition_type[:-1]
                    transition_source = regex_match.group(2)
                    transition_target = regex_match.group(3)
                    # print("Transition Line: ", line)
                    self.logger.debug(f"Transition Type: {transition_type} | Source Node: {transition_source} -> Target Node: {transition_target}")
                    transition_types.add(transition_type)
                    num_edges += 1
                    with driver.session() as session:
                        create_edge_query = """
                        MATCH (a {{rid: '{}'}}), (b {{rid: '{}'}})
                        MERGE (a)-[r:{}]->(b)
                        RETURN type(r)
                        """.format(transition_source, transition_target, transition_type)
                        self.logger.debug(f"Create Edge Query: {create_edge_query}")
                        session.run(create_edge_query)
                else:
                    self.logger.debug(f"Metadata: {line}")

            self.logger.info(f"Node Types: {node_types}")
            self.logger.info(f"Transition Types: {transition_types}")
            self.logger.info(f"Number of Nodes: {num_nodes}")
            self.logger.info(f"Number of Edges: {num_edges}")

            # Close Database Session
            driver.close()
    
    def get_file(self, container: Container, source: str, destination: Path):
        # If Destination is already a file and not directory throw an exception
        if destination.exists() and not destination.is_dir():
            self.logger.error(f"Destination must be a directory: {destination}")
            return None
        
        # Get the name of the file
        file_name = Path(source).name
        # Get Destination Path
        file_path = destination / file_name
        file_path = file_path.with_suffix(".tar")

        try:
            bits, stat = container.get_archive(source)
        # Container path doesn't exist
        except NotFound:
            self.logger.debug(f"Source path not found in container: {source}")
            return None
        # Bubble up Docker API errors as-is for caller to handle/log
        except APIError:
            self.logger.error("Docker API Returned an error")
            return None
        
        with open(file_path, 'wb') as file:
            for chunk in bits:
                file.write(chunk)
        
        self.logger.info(f"Retrieved File: {file_path}")

        return file_path
        
        
if __name__ == "__main__":
    args = parse_args()

    # Setup logging
    # create your own logger instance
    logger = configure_logging(name="GDSynth", log_level=args.log_level)

    logger.info("Initializing Metamodel Extractor")
    metamodel_extractor = MetamodelExtractor(args, logger)

    logger.info("Starting Metamodel Extraction")
    original_metamodel = metamodel_extractor.extract()

    logger.info("Initializing Metamodel Mutations")
    metamodel_mutator = MetamodelMutator(args, original_metamodel, logger)

    logger.info("Starting Metamodel Mutations")
    mutated_metamodels = metamodel_mutator.mutate()
    logger.info(f"Number of mutated metamodels: {len(mutated_metamodels)}")

    logger.info("Initializing Metamodel Adapter")
    metamodel_adapter = MetamodelAdapter(args, mutated_metamodels, logger)

    logger.info("Starting Metamodel Adaptions")
    adapted_metamodels = metamodel_adapter.adapt()

    logger.info("Initializing Test Coordinator")
    test_coordinator = TestCoordinator(args, adapted_metamodels, logger)

    logger.info("Starting Test Campaign")
    test_coordinator.start_campaign()
    logger.info("Test Campaign Running")
    test_coordinator.wait()

    logger.info("Retrieving Test Campaign Results")
    test_coordinator.get_results()

    logger.info("Test Campaign Complete")
    test_coordinator.cleanup()