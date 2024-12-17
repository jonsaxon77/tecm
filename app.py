import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.engine import URL
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.app_context().push()

    db_url = URL.create(
        "mssql+pyodbc",
        username=os.getenv("AZURE_SQL_USERNAME"),
        password=os.getenv("AZURE_SQL_PASSWORD"),
        host=os.getenv("AZURE_SQL_HOST"),
        database=os.getenv("AZURE_SQL_DATABASE"),
        query={
            "driver": "ODBC Driver 17 for SQL Server",
            "authentication": "SQLPassword",
        },
    )

    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)


    class CareItem(db.Model):
        item_id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False)
        description = db.Column(db.Text)
        age_group = db.Column(db.String(50))
        conditions = db.Column(db.String(200))

        def to_dict(self):
            return {
                'item_id': self.item_id,
                'name': self.name,
                'description': self.description,
                'age_group': self.age_group,
                'conditions': self.conditions,
            }



    CORS(app, resources={
        r"/api/recommend": {"origins": "http://localhost:5173"},
        r"/api/items": {"origins": "http://localhost:5173"},
        r"/api/items/*": {"origins": "http://localhost:5173"}
    })

    TOP_N = int(os.getenv("TOP_N", 3))
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.8))
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "care-items")

    client = AzureOpenAI(
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    # Initialize Pinecone index within the application context
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    indexes = pc.list_indexes()
    index_exists = any(index['name'] == PINECONE_INDEX_NAME for index in indexes)

    if not index_exists:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    care_item_catalog = []
    items = CareItem.query.all()
    for item in items:
        care_item_catalog.append(item.to_dict())

    upsert_items = []
    for item in care_item_catalog:
        item_id = str(item['item_id'])
        item_text = f"{item['name']} - {item['description']} (For: {item['age_group']}, Conditions: {', '.join(item['conditions'])})"
        response = client.embeddings.create(input=[item_text], model=os.getenv("EMBEDDING_DEPLOYMENT_NAME"))
        embedding = response.data[0].embedding
        upsert_items.append((item_id, embedding))

    if upsert_items:
        index.upsert(vectors=upsert_items)

    @app.route('/api/items', methods=['GET'])
    def get_items():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        items = CareItem.query.order_by(CareItem.item_id).paginate(page=page, per_page=per_page, error_out=False)
        return jsonify({
            'items': [item.to_dict() for item in items.items],
            'total_pages': items.pages
        })

    @app.route('/api/items', methods=['POST'])
    def add_item():
        data = request.get_json()
        new_item = CareItem(
            name=data['name'],
            description=data['description'],
            age_group=data['age_group'],
            conditions=data['conditions']
        )
        db.session.add(new_item)
        db.session.commit()
        item_text = f"{new_item.name} - {new_item.description} (For: {new_item.age_group}, Conditions: {', '.join(new_item.conditions)})"
        response = client.embeddings.create(input=[item_text], model=os.getenv("EMBEDDING_DEPLOYMENT_NAME"))
        embedding = response.data[0].embedding
        index.upsert(vectors=[(str(new_item.item_id), embedding)])
        return jsonify(new_item.to_dict()), 201

    @app.route('/api/items/<int:item_id>', methods=['PUT'])
    def update_item(item_id):
        item = CareItem.query.get_or_404(item_id)
        data = request.get_json()
        item.name = data['name']
        item.description = data['description']
        item.age_group = data['age_group']
        item.conditions = data['conditions']
        db.session.commit()
        item_text = f"{item.name} - {item.description} (For: {item.age_group}, Conditions: {', '.join(item.conditions)})"
        response = client.embeddings.create(input=[item_text], model=os.getenv("EMBEDDING_DEPLOYMENT_NAME"))
        embedding = response.data[0].embedding
        index.upsert(vectors=[(str(item.item_id), embedding)])
        return jsonify(item.to_dict())

    @app.route('/api/items/<int:item_id>', methods=['DELETE'])
    def delete_item(item_id):
        item = CareItem.query.get_or_404(item_id)
        db.session.delete(item)
        db.session.commit()
        index.delete(ids=[str(item_id)])
        return '', 204

    def get_care_item(item_id):
        item = db.session.get(CareItem, item_id)
        if item:
            return item.to_dict()
        return None

    @app.route('/api/recommend', methods=['POST'])
    def recommend():
        try:
            data = request.get_json()
            client_data = data.get('client')
            if not client_data:
                return jsonify({'error': 'Missing client data'}), 400
            assessment_text = f"""
               Client: {client_data['name']}
               Age: {client_data['age']}
               Conditions: {', '.join(client_data['conditions'])}
               Mobility: {client_data['mobility']}
               Living Situation: {client_data['living_situation']}
               Description: {client_data['description']}
               """
            response = client.embeddings.create(input=[assessment_text], model=os.getenv("EMBEDDING_DEPLOYMENT_NAME"))
            assessment_embedding = response.data[0].embedding

            query_results = index.query(
                vector=assessment_embedding,
                top_k=20,
                include_metadata=True
            )

            recommendations = []
            for match in query_results['matches']:
                item_id = int(match['id'])
                score = match['score']
                if score >= SCORE_THRESHOLD:
                    item = get_care_item(item_id)
                    if item:
                        # Get reasons for the match
                        reasons = get_match_reasons(assessment_text, item)
                        recommendations.append({
                            'name': item['name'],
                            'score': f"{score:.4f}",
                            'reasons': reasons  # Include reasons in the response
                        })

            return jsonify({'recommendations': recommendations})

        except Exception as e:
            print(f"An error occurred in recommend: {e}")
            return jsonify({'error': 'Failed to generate recommendations'}), 500

    def get_match_reasons(assessment_text, item):
        prompt = f"""
        You are an AI assistant helping to explain why certain care items are recommended for individuals based on their needs.

        Here's the care item: {item['name']} - {item['description'].replace('"', '\\"')} (For: {item['age_group']}, Conditions: {', '.join(item['conditions']).replace('"', '\\"')})

        Here's the client assessment: {assessment_text.replace('"', '\\"')}

        Explain why this specific care item is relevant to this specific client. Provide 3 clear and concise reasons, directly referencing details from BOTH the care item description AND the client assessment to support your explanation.
        """
        response = client.completions.create(
            model=os.getenv("COMPLETION_DEPLOYMENT_NAME"),  # Use your completion model
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        reasons = response.choices[0].text.strip().split('\n')
        return reasons

    return app

app = create_app()
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)