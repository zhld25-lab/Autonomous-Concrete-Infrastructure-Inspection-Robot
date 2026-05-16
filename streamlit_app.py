"""Streamlit demo for the autonomous concrete inspection prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from src.q_learning_agent import QLearningAgent
from src.rl_environment import InspectionEnvironment
from src.train_crack_detector import build_model
from src.visualization import plot_inspection_route


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "crack_detector.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@st.cache_resource
def load_crack_detector():
    """Load the saved crack detector if it exists."""

    if not MODEL_PATH.exists():
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model_name = checkpoint.get("model_name", "simple_cnn") if isinstance(checkpoint, dict) else "simple_cnn"
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = build_model(model_name=model_name, num_classes=2, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def predict_image(model, checkpoint, image: Image.Image):
    """Return predicted label and crack probability for one uploaded image."""

    image_size = checkpoint.get("image_size", (128, 128)) if isinstance(checkpoint, dict) else (128, 128)
    preprocess = transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        crack_probability = torch.softmax(logits, dim=1)[0, 1].item()

    label = "Crack" if crack_probability >= 0.5 else "No crack"
    return label, crack_probability


def create_risk_map(grid_size: int, random_seed: int) -> np.ndarray:
    """Create a simple simulated crack-risk map for the RL demo."""

    rng = np.random.default_rng(random_seed)
    risk_map = rng.uniform(0.05, 0.45, size=(grid_size, grid_size))

    hotspot_count = max(2, grid_size // 2)
    hotspot_indices = rng.choice(grid_size * grid_size, size=hotspot_count, replace=False)
    for index in hotspot_indices:
        row, col = divmod(int(index), grid_size)
        risk_map[row, col] = rng.uniform(0.75, 1.0)

    return risk_map


def run_planner(risk_map: np.ndarray, episodes: int, random_seed: int):
    """Train a Q-learning planner and return a learned route."""

    environment = InspectionEnvironment(risk_map, high_risk_threshold=0.65)
    agent = QLearningAgent(
        num_actions=environment.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.02,
        random_seed=random_seed,
    )
    rewards = agent.train(environment, num_episodes=episodes, progress_interval=0)
    route, actions = agent.generate_route(environment)
    return environment, rewards, route, actions


st.set_page_config(page_title="Concrete Inspection Prototype", layout="wide")

st.title("Autonomous Concrete Infrastructure Inspection Robot")
st.subheader("Vision-Based Crack Detection and RL Inspection Path Planning")

st.write(
    "This laptop-runnable prototype combines concrete crack classification, "
    "a simulated reinforcement learning inspection planner, and a ROS2-ready "
    "architecture for future robot deployment."
)

st.info(
    "Scope note: this is a prototype for perception, planning simulation, and "
    "ROS2-ready design. It is not a fully deployed robot system."
)

st.header("1. Crack Detection Demo")

model, checkpoint = load_crack_detector()
uploaded_file = st.file_uploader("Upload a concrete surface image", type=["jpg", "jpeg", "png"])

if model is None:
    st.warning(
        "No trained model was found at models/crack_detector.pt. "
        "Run notebooks/01_crack_detection_pipeline.ipynb to train the model. "
        "The RL planner demo below still works without the model."
    )
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    predicted_label, crack_probability = predict_image(model, checkpoint, image)

    col_image, col_result = st.columns([1, 1])
    with col_image:
        st.image(image, caption="Uploaded concrete image", use_container_width=True)
    with col_result:
        st.metric("Prediction", predicted_label)
        st.metric("Crack probability", f"{crack_probability:.2%}")
elif uploaded_file is None:
    st.write("Upload an image to run the crack detector.")

st.header("2. RL Inspection Planner Demo")

col_a, col_b, col_c = st.columns(3)
with col_a:
    grid_size = st.slider("Grid size", min_value=4, max_value=8, value=6)
with col_b:
    episodes = st.slider("Training episodes", min_value=100, max_value=1500, value=500, step=100)
with col_c:
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

if st.button("Run planner simulation"):
    risk_map = create_risk_map(grid_size, int(random_seed))
    environment, rewards, route, actions = run_planner(risk_map, episodes, int(random_seed))

    st.write("Simulated crack-risk map")
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(risk_map, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Crack risk")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    st.pyplot(fig)

    route_path = OUTPUT_DIR / "streamlit_inspection_route.png"
    plot_inspection_route(risk_map, route, route_path, inspected_cells=environment.inspected)
    st.image(str(route_path), caption="Learned inspection route", use_container_width=True)

    st.write("Route steps")
    st.dataframe(
        {
            "step": list(range(len(actions))),
            "action": actions,
            "position_after_action": route[1:],
        },
        use_container_width=True,
    )

    st.line_chart(rewards)

st.header("3. ROS2-Ready Architecture")

st.code(
    """
camera_node
    publishes /camera/image_raw
        v
crack_detection_node
    subscribes /camera/image_raw
    publishes /crack_detection/result
        v
rl_planner_node
    subscribes /crack_detection/result
    publishes /inspection_planner/action
        v
robot_control_node
    subscribes /inspection_planner/action
    publishes /cmd_vel
""".strip(),
    language="text",
)

st.write(
    "Future ROS2 deployment would wrap the Python modules as nodes and connect "
    "camera images, crack predictions, planner actions, and robot velocity commands."
)

