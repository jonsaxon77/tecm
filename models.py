from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

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