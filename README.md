

# AgenticTransformer: Where Transformers Meet Autonomous Intelligence

![image](https://github.com/user-attachments/assets/b56ad19d-4cf4-4b84-8123-1400a9a039aa)

Agentic Transfomer 


<img width="1728" alt="Screenshot 2025-05-26 at 17 54 52" src="https://github.com/user-attachments/assets/86609448-590d-4718-8e97-7b7d5b6d13fd" />

**Unlocking Sample-Efficient & Generalizable Policies for Autonomous Agents**

AgenticTransformer introduces a novel approach to significantly enhance the learning efficiency and generalization capabilities of Transformer-based agents. At its core is the **Cross-Episodic Curriculum (CEC)**, a powerful method that strategically structures learning experiences across multiple episodes, allowing agents to internalize complex behaviors and achieve robust policies from diverse, even sub-optimal, data.

This project goes beyond traditional Transformer-based reinforcement learning by providing a framework that enables agents to learn and adapt more effectively, especially in data-scarce and multi-task environments.

## ✨ Key Features

  * **Cross-Episodic Curriculum (CEC):** A groundbreaking method for explicitly structuring learning progression across multiple episodes, capturing policy improvement, learning progress, and expertise growth.
  * **Enhanced Sample Efficiency:** Maximizes the utility of limited and varied quality data, making it ideal for real-world scenarios where data collection is expensive (e.g., robotics).
  * **Superior Generalization:** Trains agents to acquire more robust and adaptable policies that perform exceptionally well in unseen tasks, environments, and out-of-distribution scenarios.
  * **Robust Learning from Sub-Optimal Data:** Unlike some methods that struggle with imperfect data, AgenticTransformer leverages a "chain of hindsight" to effectively learn from mixed-quality or sub-optimal trajectories.
  * **Scalability:** Demonstrates promising scaling trends, with larger models consistently yielding improved results.
  * **Applications:** Proven effectiveness in Multi-Task Reinforcement Learning (e.g., DeepMind Lab) and Imitation Learning from mixed-quality human demonstrations (e.g., robotics).

## 🚀 Why AgenticTransformer?

Traditional Transformer-based policies, while powerful, often face challenges in efficiently extracting actionable insights from large, diverse datasets, especially when dealing with sub-optimal or evolving experiences. AgenticTransformer tackles this head-on by:

  * **Structuring Knowledge:** Instead of merely processing sequences, it intelligently organizes "cross-episodic experiences" into a meaningful curriculum.
  * **Learning Progression:** It enables the Transformer to understand *how* a policy improves over time or *how* an expert's skill develops, rather than just observing isolated actions.
  * **Closing the Gap:** It bridges the gap between state-of-the-art Transformer architectures and the practical demands of real-world autonomous intelligence, where data is often imperfect and generalization is paramount.

## 📚 How It Works (High-Level)

The core mechanism revolves around the **Cross-Episodic Curriculum (CEC)**, which operates in two main stages:

1.  **Preparation of Curricular Data:**

      * Multiple learning experiences (trajectories) are collected and ordered to capture a progression (e.g., an agent getting better at a task, an expert demonstrating increasing proficiency, or an agent completing progressively harder environments).
      * This ordering forms the "curriculum" that provides explicit signals about improvement or learning progress.

2.  **Model Training with Cross-Episodic Attention:**

      * During training, the Transformer model is equipped with a novel attention mechanism that allows it to "look back" beyond the current episode and leverage the organized curricular data.
      * This "cross-episodic attention" enables the model to trace back improved behaviors encoded in the curriculum, allowing for more efficient policy internalisation and refinement.

For a deeper dive into the technical details, please refer to our upcoming research paper (link to be added).

## 🛠️ Installation

To get started with AgenticTransformer, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/praneethk-ai/AgenticTransfomer-Where-Transformers-Meet-Autonomous-Intelligence.git
    cd AgenticTransfomer-Where-Transformers-Meet-Autonomous-Intelligence
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Ensure `requirements.txt` is present and lists all necessary libraries like PyTorch, Transformers, Gymnasium, etc.)*

## 🚀 Getting Started (Usage Examples)

*(This section will require placeholder code as the actual implementation details are not yet visible. I'll provide illustrative examples.)*

We provide examples to demonstrate how to prepare data for CEC and train an AgenticTransformer.

### Example 1: Training an Agent in DeepMind Lab (Multi-task RL)

```python
# Coming Soon: Example script for training in DeepMind Lab
# This will involve:
# 1. Setting up DeepMind Lab environment (might require specific installations/builds)
# 2. Collecting or loading multi-episode trajectories
# 3. Applying the Cross-Episodic Curriculum (CEC) data preparation
# 4. Initializing and training the AgenticTransformer model
# from agentic_transformer.environments import DeepMindLabEnv
# from agentic_transformer.data_curriculum import create_cec_dataset
# from agentic_transformer.models import AgenticTransformerPolicy
# from agentic_transformer.train import train_agent

# env = DeepMindLabEnv("seek_and_find_humanoid")
# raw_trajectories = collect_raw_data(env, num_episodes=100) # placeholder
# cec_dataset = create_cec_dataset(raw_trajectories, curriculum_strategy="policy_improvement")

# model = AgenticTransformerPolicy(config)
# train_agent(model, cec_dataset)
```

### Example 2: Imitation Learning from Mixed-Quality Demonstrations

```python
# Coming Soon: Example script for imitation learning
# This will involve:
# 1. Loading demonstrations with varying expertise levels
# 2. Applying the Cross-Episodic Curriculum (CEC) to order demonstrations by improving skill
# 3. Training an AgenticTransformer to imitate the expert's progression
# from agentic_transformer.data_curriculum import create_cec_dataset
# from agentic_transformer.models import AgenticTransformerPolicy
# from agentic_transformer.train import train_il_agent

# raw_demos = load_mixed_quality_demonstrations("robot_arm_tasks.pkl") # placeholder
# cec_demos = create_cec_dataset(raw_demos, curriculum_strategy="expertise_growth")

# model = AgenticTransformerPolicy(config)
# train_il_agent(model, cec_demos)
```


