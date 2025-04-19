import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Random Forest Classifier", layout="wide")

st.title("🌲 Random Forest Classifier web app")

# Sidebar for parameter tuning
st.sidebar.header("🔧 Hyperparameters")

n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 1, 300, 100)
max_features = st.sidebar.selectbox("Max Features", ['sqrt', 'log2'])
max_samples = st.sidebar.slider("Max Samples", 0.1, 1.0, 0.8)
max_depth = st.sidebar.slider("Max Depth", 1, 50, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
bootstrap = st.sidebar.selectbox("Bootstrap Samples", [True, False])
criterion = st.sidebar.selectbox("Criterion", ['gini', 'entropy', 'log_loss'])
random_state = st.sidebar.number_input("Random State", value=42)

# Generate concentric circles data

np.random.seed(42)
X,y = make_circles(n_samples=500,factor=0.1, noise=0.35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=random_state)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# Scale features    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Button to run training
run_model = st.button("▶️ Run Algorithm")

if run_model:
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features = max_features,
        max_samples=max_samples,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    st.success(f"✅ Model Trained! Accuracy: {acc:.2f}")

    # Plot decision boundary
    def plot_decision_boundary(X, y, model, title="RF Decision Boundary"):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
        plt.title(title)
        st.pyplot(plt)

    plot_decision_boundary(X_train_scaled, y_train, model, "RF Decision Boundary (Train Set)")

else:
    st.info("👈 Adjust the hyperparameters and click 'Run Algorithm' to start training.")
