import os
import pytest
import pandas as pd
import numpy as np
from pydantic import BaseModel

from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agent import app as adk_app
from app.agent import WorkflowInput


@pytest.fixture
def generate_mock_dataset():
    # Create directory if not exists
    os.makedirs("uploads", exist_ok=True)
    
    # Generate 12000 rows to trigger Latin Hypercube sampling (limit is > 10k)
    np.random.seed(42)
    n_samples = 12000
    f1 = np.random.uniform(10, 50, n_samples)
    f2 = np.random.uniform(1, 10, n_samples)
    f3 = np.random.uniform(100, 200, n_samples)
    
    # Target = linear combination + noise
    target = f1 * 2.5 + f2 * 10 - f3 * 0.5 + np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame({
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "Target": target
    })
    
    file_path = "uploads/mock_housing.csv"
    df.to_csv(file_path, index=False)
    
    yield file_path
    
    # Clean up files created during test
    for f in [file_path, "sampled_dataset.csv", "models/rf_model.pkl", 
              "images/train_predicted_vs_target.png", "images/test_predicted_vs_target.png",
              "images/shap_beeswarm.png", "images/shap_feature_importance.png"]:
        if os.path.exists(f):
            os.remove(f)


@pytest.mark.asyncio
async def test_dice_counterfactuals_workflow(generate_mock_dataset):
    file_path = generate_mock_dataset
    
    # Prepare query instance for DiCE
    query_instance = {"F1": 25.0, "F2": 5.0, "F3": 150.0}
    target_range = [2.0, 10.0]  # Try to reach a target range
    
    input_payload = WorkflowInput(
        file_path=file_path,
        target_column="Target",
        query_instance=query_instance,
        target_range=target_range
    )
    
    runner = InMemoryRunner(app=adk_app)
    session = await runner.session_service.create_session(
        app_name=adk_app.name, user_id="test_user"
    )
    
    # Run workflow
    output_event = None
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_payload.model_dump_json())]
        )
    ):
        if event.output is not None:
            output_event = event.output
            
    assert output_event is not None
    assert output_event["validation"]["status"] == "success"
    assert output_event["validation"]["num_rows_sampled"] == 3000
    assert output_event["training"]["status"] == "success"
    assert output_event["shap"]["status"] == "success"
    assert output_event["dice"]["status"] == "success"
    
    # Check generated files
    assert os.path.exists("models/rf_model.pkl")
    assert os.path.exists("images/train_predicted_vs_target.png")
    assert os.path.exists("images/test_predicted_vs_target.png")
    assert os.path.exists("images/shap_beeswarm.png")
    assert os.path.exists("images/shap_feature_importance.png")
