# Running This Project in PyCharm

This guide is for running the full project locally in PyCharm.

## 1. Open the Project

Open this folder in PyCharm:

```text
Autonomous-Concrete-Infrastructure-Inspection-Robot
```

Make sure PyCharm opens the folder that contains:

```text
README.md
requirements.txt
run_all.py
src/
notebooks/
outputs/
models/
```

## 2. Configure Python Interpreter

In PyCharm:

```text
File -> Settings -> Project -> Python Interpreter
```

Choose your Python interpreter. Python 3.10 or newer is recommended.

## 3. Install Dependencies

Open the PyCharm Terminal and run:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install the CPU version:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 4. Check Dataset Location

The dataset must be placed here:

```text
data/raw/concrete_crack/
```

Supported structures:

```text
data/raw/concrete_crack/Positive/
data/raw/concrete_crack/Negative/
```

or:

```text
data/raw/concrete_crack/crack/
data/raw/concrete_crack/no_crack/
```

## 5. Run the Full Project

In PyCharm, open:

```text
run_all.py
```

Right-click inside the file and choose:

```text
Run 'run_all'
```

This will run:

- Crack image loading
- Class distribution summary
- Sample image visualization
- Train/validation/test split
- CNN training
- Model evaluation
- Confusion matrix generation
- Prediction examples
- Q-learning inspection planner
- RL route visualization

## 6. Expected Outputs

After running `run_all.py`, check:

```text
outputs/
models/
```

Expected files:

```text
outputs/class_distribution.csv
outputs/sample_images.png
outputs/training_history.csv
outputs/training_history.png
outputs/classification_report.txt
outputs/confusion_matrix.png
outputs/prediction_examples.png
outputs/crack_risk_map.png
outputs/rl_training_rewards.png
outputs/inspection_route.png
models/crack_detector.pt
```

## 7. Optional: Run Streamlit App

After the model exists, run:

```bash
streamlit run streamlit_app.py
```

The app lets you upload a concrete image and run the RL simulation demo.

## 8. Optional: Run Notebooks

If your PyCharm version supports notebooks, you can also run:

```text
notebooks/01_crack_detection_pipeline.ipynb
notebooks/02_rl_inspection_planner.ipynb
```

If not, use Jupyter:

```bash
jupyter notebook
```

