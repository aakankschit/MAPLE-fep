from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from ..dataset.base_dataset import BaseDataset


class GraphSetup:
    """
    A class to create and visualize networkx graphs from FEP datasets.

    This class takes a dataset (inheriting from BaseDataset) and creates a networkx graph
    with edge attributes including FEP values, experimental values, CCC values, and MAP values.
    It provides methods to visualize the graph in a circular layout.

    Attributes
    ----------
    dataset : BaseDataset
        The input dataset containing node and edge information.
    graph : nx.DiGraph
        The networkx directed graph representation of the dataset.
    node_mapping : Tuple[Dict[str, int], Dict[int, str]]
        Node-to-index and index-to-node mappings.
    edge_attributes : Dict[str, List]
        Dictionary containing edge attributes (FEP, experimental, CCC, MAP).

    Example
    -------
        # Create dataset
        dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

        # Create graph setup
        graph_setup = GraphSetup(dataset)

        # Visualize the graph
        graph_setup.visualize_graph(title="CDK8 FEP Network")
    """

    def __init__(self, dataset: BaseDataset):
        """
        Initialize the GraphSetup with a dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset object containing node and edge information.

        Raises
        ------
        ValueError
            If the dataset doesn't have the required attributes or methods.
        """
        self.dataset = dataset
        self.node_mapping = dataset.get_node_mapping()
        self.edge_attributes = {}

        # Create the networkx graph
        self.graph = self._create_graph()

    def _create_graph(self) -> nx.DiGraph:
        """
        Create a networkx directed graph from the dataset.
        
        Uses numeric indices from dataset_edges for consistency with model inference.

        Returns
        -------
        nx.DiGraph
            Directed graph with nodes and edges from the dataset.
        """
        # Get dataframes - dataset_edges has all the data we need
        edges_df, nodes_df = self.dataset.get_dataframes()

        # Create directed graph
        graph = nx.DiGraph()

        # Add all unique nodes from edges (using numeric indices)
        all_nodes = set(edges_df["Source"].unique()) | set(edges_df["Destination"].unique())
        for node in all_nodes:
            graph.add_node(node)

        # Add edges with attributes from dataset_edges
        for idx, row in edges_df.iterrows():
            source_idx = row["Source"]
            dest_idx = row["Destination"]
            
            # Extract all relevant columns as edge attributes
            edge_attrs = {}
            for col in edges_df.columns:
                if col not in ["Source", "Destination"] and pd.notna(row[col]):
                    edge_attrs[col] = row[col]
            
            # Add edge with attributes
            graph.add_edge(source_idx, dest_idx, **edge_attrs)

        return graph

    def _extract_edge_attributes_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract edge attributes directly from a DataFrame row.
        
        Parameters
        ----------
        row : pd.Series
            A row from the edges DataFrame
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of edge attributes
        """
        attributes = {}
        
        # Map column names to standardized attribute names
        # These are the columns we want to include as edge attributes
        column_mapping = {
            # Main values
            "DeltaDeltaG": "DeltaDeltaG",
            "CCC": "CCC", 
            "MAP": "MAP",
            "VI": "VI",
            # Experimental values
            "Experimental DeltaDeltaG": "Experimental DeltaDeltaG",
            "Experimental DeltaG": "Experimental DeltaDeltaG",
            "Exp.": "Experimental DeltaDeltaG",
            # Error values
            "DeltaDeltaG Error": "DeltaDeltaG Error",
            "CCC Error": "CCC Error",
            "MAP_uncertainty": "MAP_uncertainty",
            "VI_uncertainty": "VI_uncertainty",
        }
        
        # Extract available attributes from the row
        for col_name, attr_name in column_mapping.items():
            if col_name in row.index and pd.notna(row[col_name]):
                attributes[attr_name] = float(row[col_name])
                
        return attributes
    
    def _extract_edge_attributes(
        self, edge_idx: int, graph_data: Dict[str, Any], edges_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract edge attributes from the dataset using standardized column names.

        Parameters
        ----------
        edge_idx : int
            Index of the edge in the graph data.
        graph_data : Dict[str, Any]
            Graph data dictionary.
        edges_df : pd.DataFrame
            Edge DataFrame.

        Returns
        -------
        Dict[str, Any]
            Dictionary of edge attributes using standardized names.
        """
        attributes = {}

        # Extract FEP value (using standardized names)
        if "FEP" in graph_data and edge_idx < len(graph_data["FEP"]):
            attributes["DeltaDeltaG"] = graph_data["FEP"][edge_idx]

        # Extract CCC value
        if "CCC" in graph_data and edge_idx < len(graph_data["CCC"]):
            attributes["CCC"] = graph_data["CCC"][edge_idx]

        # Extract experimental values from edges DataFrame
        source_idx = graph_data["src"][edge_idx]
        dest_idx = graph_data["dst"][edge_idx]
        source_node = self.node_mapping[1][source_idx]
        dest_node = self.node_mapping[1][dest_idx]

        # Find the row in edges_df that matches this edge
        # Handle different column name conventions (standardized and legacy)
        edge_row = None

        # Check for standardized column names first
        if "Source" in edges_df.columns and "Destination" in edges_df.columns:
            # After _apply_node_mapping, Source/Destination contain numeric indices
            # So we need to compare with the numeric indices, not node names
            edge_row = edges_df[
                (edges_df["Source"] == source_idx)
                & (edges_df["Destination"] == dest_idx)
            ]
        # Fallback to legacy column names
        elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
            edge_row = edges_df[
                (edges_df["Ligand1"] == source_idx)
                & (edges_df["Ligand2"] == dest_idx)
            ]

        if edge_row is not None and not edge_row.empty:
            # Handle experimental values (standardized names first)
            if "Experimental DeltaDeltaG" in edge_row.columns:
                attributes["Experimental DeltaDeltaG"] = edge_row[
                    "Experimental DeltaDeltaG"
                ].iloc[0]
            elif "Experimental DeltaG" in edge_row.columns:
                attributes["Experimental DeltaG"] = edge_row[
                    "Experimental DeltaG"
                ].iloc[0]
            elif "Exp." in edge_row.columns:
                attributes["Experimental DeltaDeltaG"] = edge_row["Exp."].iloc[0]

            # Handle FEP/DeltaDeltaG values from edges DataFrame (standardized names first)
            if "DeltaDeltaG" in edge_row.columns:
                attributes["DeltaDeltaG"] = edge_row["DeltaDeltaG"].iloc[0]
            elif "DeltaG" in edge_row.columns:
                attributes["DeltaDeltaG"] = edge_row["DeltaG"].iloc[0]
            elif "FEP" in edge_row.columns:
                attributes["DeltaDeltaG"] = edge_row["FEP"].iloc[0]

            # Handle error values if available
            if "DeltaDeltaG Error" in edge_row.columns:
                attributes["DeltaDeltaG Error"] = edge_row["DeltaDeltaG Error"].iloc[0]
            elif "DeltaG Error" in edge_row.columns:
                attributes["DeltaDeltaG Error"] = edge_row["DeltaG Error"].iloc[0]
            elif "FEP Error" in edge_row.columns:
                attributes["DeltaDeltaG Error"] = edge_row["FEP Error"].iloc[0]

            # Handle CCC error values if available
            if "CCC Error" in edge_row.columns:
                attributes["CCC Error"] = edge_row["CCC Error"].iloc[0]

            # Handle MAP values if available (from edge DataFrame)
            if "MAP" in edge_row.columns:
                attributes["MAP"] = edge_row["MAP"].iloc[0]

            # Handle VI values if available (from edge DataFrame)
            if "VI" in edge_row.columns:
                attributes["VI"] = edge_row["VI"].iloc[0]

        # Extract MAP values if available (from estimator results as fallback)
        if "MAP" not in attributes:
            estimators = self.dataset.get_estimators()
            if estimators:
                # Get the latest estimator results
                latest_estimator = estimators[-1]
                if (
                    isinstance(latest_estimator, dict)
                    and "edge_estimates" in latest_estimator
                ):
                    edge_estimates = latest_estimator["edge_estimates"]
                    if edge_idx < len(edge_estimates):
                        attributes["MAP"] = edge_estimates[edge_idx]

        return attributes

    def visualize_graph(
        self,
        title: str = "FEP Network Graph",
        figsize: Tuple[int, int] = (12, 12),
        node_size: int = 1000,
        node_color: str = "lightblue",
        edge_color: str = "gray",
        edge_width: float = 1.0,
        font_size: int = 10,
        show_edge_labels: bool = True,
        edge_label_attr: str = "DeltaDeltaG",
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """
        Visualize the graph in a circular layout.

        Parameters
        ----------
        title : str, default="FEP Network Graph"
            Title for the plot.
        figsize : Tuple[int, int], default=(12, 12)
            Figure size (width, height).
        node_size : int, default=1000
            Size of nodes in the plot.
        node_color : str, default='lightblue'
            Color of nodes.
        edge_color : str, default='gray'
            Color of edges.
        edge_width : float, default=1.0
            Width of edges.
        font_size : int, default=10
            Font size for labels.
        show_edge_labels : bool, default=True
            Whether to show edge labels.
        edge_label_attr : str, default='DeltaDeltaG'
            Edge attribute to use for labels (standardized names: 'DeltaDeltaG', 'Experimental DeltaDeltaG', 'CCC').
        save_path : Optional[str], default=None
            Path to save the plot. If None, plot is displayed.
        dpi : int, default=300
            DPI for saved plots.
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate circular layout
        pos = nx.circular_layout(self.graph)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, node_size=node_size, node_color=node_color, ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color=edge_color,
            width=edge_width,
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            ax=ax,
        )

        # Get idx2node mapping for displaying original names
        _, idx2node = self.node_mapping
        
        # Create labels using original node names
        node_labels = {}
        for node in self.graph.nodes():
            # If node is a numeric index, convert to original name for display
            if node in idx2node:
                node_labels[node] = str(idx2node[node])
            else:
                # Fallback to using the node itself as label
                node_labels[node] = str(node)
        
        # Draw node labels with original names
        nx.draw_networkx_labels(
            self.graph, pos, labels=node_labels, font_size=font_size, font_weight="bold", ax=ax
        )

        # Draw edge labels if requested
        if show_edge_labels:
            edge_labels = {}
            for u, v, data in self.graph.edges(data=True):
                if edge_label_attr in data:
                    # Format the label based on the attribute type
                    value = data[edge_label_attr]
                    if isinstance(value, (int, float)):
                        edge_labels[(u, v)] = f"{value:.2f}"
                    else:
                        edge_labels[(u, v)] = str(value)

            nx.draw_networkx_edge_labels(
                self.graph, pos, edge_labels=edge_labels, font_size=font_size - 2, ax=ax
            )

        # Set title and remove axes
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")
        ax.axis("off")

        # Adjust layout
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Graph saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the graph.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing graph information:
            - 'num_nodes': Number of nodes
            - 'num_edges': Number of edges
            - 'node_names': List of node names
            - 'edge_attributes': Available edge attributes
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "node_names": list(self.graph.nodes()),
            "edge_attributes": list(
                set(
                    [
                        attr
                        for _, _, data in self.graph.edges(data=True)
                        for attr in data.keys()
                    ]
                )
            ),
        }

    def get_edge_dataframe(self) -> pd.DataFrame:
        """
        Get edge data as a pandas DataFrame with standardized column names.

        Returns
        -------
        pd.DataFrame
            DataFrame containing edge information with standardized column names:
            - 'Source': Source node identifier
            - 'Destination': Destination node identifier
            - 'DeltaDeltaG': Relative free energy values
            - 'Experimental DeltaDeltaG': Experimental relative free energy values (if available)
            - 'CCC': Cycle closure corrected values (if available)
            - 'DeltaDeltaG Error': Relative free energy uncertainty (if available)
            - 'CCC Error': CCC uncertainty (if available)
            - 'MAP': MAP estimates (if available)
        """
        edge_data = []
        for u, v, data in self.graph.edges(data=True):
            row = {"Source": u, "Destination": v, **data}
            edge_data.append(row)

        return pd.DataFrame(edge_data)

    def get_node_dataframe(self) -> pd.DataFrame:
        """
        Get node data as a pandas DataFrame with standardized column names.

        Returns
        -------
        pd.DataFrame
            DataFrame containing node information with standardized column names:
            - 'Name': Node/ligand identifier
            - 'degree': Total degree (in + out)
            - 'in_degree': Incoming degree
            - 'out_degree': Outgoing degree
        """
        node_data = []
        for node in self.graph.nodes():
            row = {
                "Name": node,
                "degree": self.graph.degree(node),
                "in_degree": self.graph.in_degree(node),
                "out_degree": self.graph.out_degree(node),
            }
            node_data.append(row)

        return pd.DataFrame(node_data)
