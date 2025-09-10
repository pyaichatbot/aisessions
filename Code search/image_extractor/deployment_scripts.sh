#!/bin/bash
# deploy.sh - Complete deployment script

set -e

echo "🚀 Deploying Enterprise Architecture Diagram Extractor..."

# Create necessary directories
mkdir -p input output logs notebooks

# Create .env file with default values
cat > .env << EOF
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DB=neo4j

# Application Configuration
PYTHONPATH=/app
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
LOG_LEVEL=INFO

# Optional: API Configuration
FLASK_ENV=production
API_PORT=8000
EOF

echo "📁 Created project directories and configuration files"

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build --no-cache

echo "🚀 Starting services..."
docker-compose up -d

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
until docker-compose exec neo4j cypher-shell -u neo4j -p  "RETURN 1" > /dev/null 2>&1; do
    echo "Still waiting for Neo4j..."
    sleep 10
done

echo "✅ Neo4j is ready!"

# Install additional packages in the extractor container
echo "📦 Installing additional packages..."
docker-compose exec diagram_extractor pip install jupyterlab flask flask-cors gunicorn

echo "🎉 Deployment completed successfully!"
echo ""
echo "🔗 Service URLs:"
echo "   Neo4j Browser: http://localhost:7474 (neo4j/)"
echo "   Jupyter Lab:   http://localhost:8888"
echo "   API Service:   http://localhost:8000"
echo ""
echo "📝 Usage Examples:"
echo "   # Extract from image:"
echo "   docker-compose exec diagram_extractor python3 enterprise_extractor.py input/diagram.png -o output/result"
echo ""
echo "   # Run original agent:"
echo "   docker-compose exec diagram_extractor python3 agent.py run input/diagram.mmd --namespace production"
echo ""
echo "   # Access container shell:"
echo "   docker-compose exec diagram_extractor bash"

# Create usage examples script
cat > usage_examples.sh << 'EOF'
#!/bin/bash
# usage_examples.sh - Example commands for the diagram extractor

echo "📋 Enterprise Architecture Diagram Extractor - Usage Examples"
echo "============================================================"

echo ""
echo "1️⃣  Extract architecture from image with visualization:"
echo "   docker-compose exec diagram_extractor python3 enterprise_extractor.py input/architecture.png --output output/arch_results --visualize"

echo ""
echo "2️⃣  Process Mermaid diagram with original agent:"
echo "   docker-compose exec diagram_extractor python3 agent.py run input/system.mmd --namespace production"

echo ""
echo "3️⃣  Extract only (no Neo4j push):"
echo "   docker-compose exec diagram_extractor python3 enterprise_extractor.py input/diagram.png --output output/extracted --format json"

echo ""
echo "4️⃣  Generate Cypher statements:"
echo "   docker-compose exec diagram_extractor python3 enterprise_extractor.py input/network.png --output output/cypher --format cypher"

echo ""
echo "5️⃣  Copy files to container:"
echo "   docker cp your_diagram.png \$(docker-compose ps -q diagram_extractor):/app/input/"

echo ""
echo "6️⃣  Get results from container:"
echo "   docker cp \$(docker-compose ps -q diagram_extractor):/app/output/. ./local_output/"

echo ""
echo "7️⃣  View Neo4j data:"
echo "   Open http://localhost:7474 in browser"
echo "   Username: neo4j, Password: "
echo "   Query: MATCH (n) RETURN n LIMIT 25"

echo ""
echo "8️⃣  Access Jupyter for analysis:"
echo "   Open http://localhost:8888"
echo "   Create new notebook to analyze extracted graphs"

echo ""
echo "9️⃣  Check container logs:"
echo "   docker-compose logs -f diagram_extractor"

echo ""
echo "🔟  Scale processing (if needed):"
echo "   docker-compose up -d --scale diagram_extractor=3"
EOF

chmod +x usage_examples.sh

# Create API example script
cat > api_example.py << 'EOF'
"""
api.py - Simple Flask API for diagram extraction
"""
import os
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path

# Import our extractor (assuming it's in the same directory)
try:
    from enterprise_extractor import EnterpriseArchitectureExtractor
except ImportError:
    # Fallback for development
    EnterpriseArchitectureExtractor = None

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/app/input'
OUTPUT_FOLDER = '/app/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "diagram-extractor-api"})

@app.route('/extract', methods=['POST'])
def extract_diagram():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported"}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract knowledge graph
        if EnterpriseArchitectureExtractor:
            extractor = EnterpriseArchitectureExtractor()
            kg = extractor.extract_from_image_path(filepath)
            
            # Save results
            output_path = os.path.join(OUTPUT_FOLDER, f"{Path(filename).stem}_results.json")
            extractor.save_results(kg, output_path.replace('.json', ''), format='json')
            
            # Return summary
            return jsonify({
                "status": "success",
                "summary": {
                    "components": len(kg.nodes),
                    "relationships": len(kg.relationships),
                    "component_types": list(set(node.type for node in kg.nodes))
                },
                "results_file": output_path
            })
        else:
            return jsonify({"error": "Extractor not available"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    try:
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
EOF

# Create Jupyter analysis notebook
mkdir -p notebooks
cat > notebooks/architecture_analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Enterprise Architecture Analysis\n",
    "\n",
    "This notebook demonstrates how to analyze extracted architecture diagrams using the knowledge graph data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "from neo4j import GraphDatabase\n",
    "import seaborn as sns\n",
    "\n",
    "# Neo4j connection\n",
    "driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', ''))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Extracted Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load JSON results\n",
    "with open('../output/results.json', 'r') as f:\n",
    "    kg_data = json.load(f)\n",
    "\n",
    "print(f\"Components: {len(kg_data['nodes'])}\")\n",
    "print(f\"Relationships: {len(kg_data['relationships'])}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Component Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analyze component types\n",
    "component_types = [node['type'] for node in kg_data['nodes']]\n",
    "type_counts = pd.Series(component_types).value_counts()\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "type_counts.plot(kind='bar')\n",
    "plt.title('Component Types Distribution')\n",
    "plt.xticks(rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Network Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create NetworkX graph\n",
    "G = nx.DiGraph()\n",
    "\n",
    "# Add nodes\n",
    "for node in kg_data['nodes']:\n",
    "    G.add_node(node['id'], **node)\n",
    "\n",
    "# Add edges\n",
    "for rel in kg_data['relationships']:\n",
    "    G.add_edge(rel['source_id'], rel['target_id'], **rel)\n",
    "\n",
    "print(f\"Graph density: {nx.density(G):.3f}\")\n",
    "print(f\"Number of connected components: {nx.number_weakly_connected_components(G)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "version": "3.8+"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "📋 Created additional files:"
echo "   - usage_examples.sh (run: ./usage_examples.sh)"
echo "   - api_example.py (Flask API for web interface)"
echo "   - notebooks/architecture_analysis.ipynb (Jupyter analysis)"
echo ""
echo "🔧 Quick Start Commands:"
echo "   ./deploy.sh                    # Deploy everything"
echo "   ./usage_examples.sh            # Show usage examples"
echo "   docker-compose logs -f         # View all logs"
echo "   docker-compose down            # Stop all services"
EOF