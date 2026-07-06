# Copyright (c) 2026 MyCompany LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from google.adk.workflow import Workflow, JoinNode, START
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.adk.apps import App

from . import tools

# 1. Define Pydantic schemas for Workflow Inputs and Outputs
class WorkflowInput(BaseModel):
    file_path: str = Field(description="Path to the uploaded dataset CSV or Excel file.")
    target_column: str = Field(description="Name of the target variable/column.")
    query_instance: Dict[str, Any] = Field(description="Feature values for the query instance to explain.")
    target_range: List[float] = Field(description="Desired range [lower, upper] for the counterfactual target.")
    permitted_ranges: Optional[Dict[str, List[float]]] = Field(
        default=None, 
        description="Optional dictionary specifying permitted [min, max] range for features."
    )

class ValidationResult(BaseModel):
    status: str
    num_rows_original: int
    num_rows_sampled: int
    features: List[str]
    target: str
    sampled_path: str

class TrainingResult(BaseModel):
    status: str
    train_r2: float
    test_r2: float
    model_path: str
    train_plot_path: str
    test_plot_path: str

class ShapResult(BaseModel):
    status: str
    beeswarm_path: str
    feature_importance_path: str

class DiceResult(BaseModel):
    status: str
    counterfactuals: List[Dict[str, Any]]

class WorkflowOutput(BaseModel):
    validation: ValidationResult
    training: TrainingResult
    shap: ShapResult
    dice: DiceResult


# 2. Define Workflow Nodes (as Functions)
def validate_node(ctx: Context, node_input: WorkflowInput) -> Event:
    """Validate and sample the dataset."""
    result = tools.validate_and_sample_dataset(node_input.file_path, node_input.target_column)
    if result["status"] == "error":
        raise ValueError(result["message"])
    
    # Store key parameters in workflow state for downstream nodes
    state_delta = {
        "target_column": node_input.target_column,
        "query_instance": node_input.query_instance,
        "target_range": node_input.target_range,
        "permitted_ranges": node_input.permitted_ranges,
        "sampled_path": result["sampled_path"]
    }
    return Event(output=result, state=state_delta)


def train_node(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """Train the model and save plots."""
    sampled_path = ctx.state["sampled_path"]
    target_column = ctx.state["target_column"]
    
    result = tools.train_rf_quantile_regressor(sampled_path, target_column)
    if result["status"] == "error":
        raise ValueError("Model training failed.")
        
    state_delta = {
        "model_path": result["model_path"]
    }
    return Event(output=result, state=state_delta)


def shap_node(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """Run SHAP analysis in parallel."""
    model_path = ctx.state["model_path"]
    sampled_path = ctx.state["sampled_path"]
    target_column = ctx.state["target_column"]
    
    result = tools.generate_shap_analysis(model_path, sampled_path, target_column)
    return Event(output=result)


def dice_node(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """Generate DiCE counterfactuals in parallel."""
    model_path = ctx.state["model_path"]
    sampled_path = ctx.state["sampled_path"]
    target_column = ctx.state["target_column"]
    query_instance = ctx.state["query_instance"]
    target_range = ctx.state["target_range"]
    permitted_ranges = ctx.state.get("permitted_ranges")
    
    result = tools.generate_dice_counterfactuals(
        model_path, sampled_path, target_column, query_instance, target_range, permitted_ranges
    )
    return Event(output=result)


def combiner_node(ctx: Context, node_input: Dict[str, Any]) -> WorkflowOutput:
    """Combine outputs from the parallel execution branches."""
    # node_input from JoinNode is a dict keyed by predecessor names:
    # {"shap_node": shap_result, "dice_node": dice_result}
    # We retrieve the validation and training outputs from state or predecessors
    validation_res = ctx.state["sampled_path"] # we can reconstruct or get the raw outputs stored in history/state
    # Let's save validation and training outputs in the state for ease of extraction
    
    validation_dict = ctx.state.get("validation_output")
    training_dict = ctx.state.get("training_output")
    
    # Alternatively, since we didn't save the full dicts, let's grab them
    return WorkflowOutput(
        validation=ValidationResult(**ctx.state["validation_output"]),
        training=TrainingResult(**ctx.state["training_output"]),
        shap=ShapResult(**node_input["shap_node"]),
        dice=DiceResult(**node_input["dice_node"])
    )


# Adjusting node definitions to record outputs to state before proceeding
def validate_node_wrapper(ctx: Context, node_input: WorkflowInput) -> Event:
    ev = validate_node(ctx, node_input)
    ev.actions.state_delta["validation_output"] = ev.output
    return ev

def train_node_wrapper(ctx: Context, node_input: Dict[str, Any]) -> Event:
    ev = train_node(ctx, node_input)
    ev.actions.state_delta["training_output"] = ev.output
    return ev


# 3. Build the graph edges
join = JoinNode(name="merge")

edges = [
    (START, validate_node_wrapper),
    (validate_node_wrapper, train_node_wrapper),
    # Fan out to SHAP and DiCE in parallel
    (train_node_wrapper, (shap_node, dice_node)),
    # Fan in to JoinNode
    ((shap_node, dice_node), join),
    # Combine outputs
    (join, combiner_node)
]

root_agent = Workflow(
    name="dice_counterfactuals_workflow",
    edges=edges,
    input_schema=WorkflowInput,
    output_schema=WorkflowOutput,
    description="A workflow research assistant that validates dataset upload, trains model, and generates SHAP plots and DiCE counterfactuals in parallel."
)

app = App(
    root_agent=root_agent,
    name="dice_counterfactuals_agent",
)
