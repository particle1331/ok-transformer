from fastapi import FastAPI, Query

app = FastAPI()

# Sample data
items = [
    {"id": 1, "name": "Item 1", "description": "Description 1", "price": 10.0},
    {"id": 2, "name": "Item 2", "description": "Description 2", "price": 20.0},
]

@app.get("/items")
def read_items(fields: list[str] = Query(None)):
    if not fields:
        return items
    
    result = []
    for item in items:
        filtered_item = {field: item[field] for field in fields if field in item}
        result.append(filtered_item)
    
    return result
