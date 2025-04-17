from fastapi import FastAPI
import uvicorn
from blog_api import blog_router

app = FastAPI()

app.include_router(blog_router)

@app.get("/")
def default():
    return {"Message" : "Welcome to Blog Writer !","status":"200"}


if __name__ == "__main__":
    uvicorn.run(app,port=8000,reload=True) 