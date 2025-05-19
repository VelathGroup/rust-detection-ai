# Python-related
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Model files
*.pt
*.pth
*.onnx
*.tflite
*.pb

# Dataset and results
data/*/images/*
data/*/labels/*
data/*/masks/*
sample_data/*
results/*

# Exceptions: keep directory structure
!data/*/images/.gitkeep
!data/*/labels/.gitkeep
!data/*/masks/.gitkeep
!sample_data/.gitkeep
!results/.gitkeep
!models/.gitkeep

# IDE related
.idea/
.vscode/
*.swp
*.swo

# OS related
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Training logs
runs/
wandb/
logs/
