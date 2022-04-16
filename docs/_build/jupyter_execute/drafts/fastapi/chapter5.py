#!/usr/bin/env python
# coding: utf-8

# # Dependency Injections

# ```{admonition} Attribution
# This notebook follows Chapter 5: *Dependency Injections in FastAPI* of {cite}`Voron2021`. Source files for running the background local servers can be found [here](https://github.com/particle1331/machine-learning/tree/master/docs/notebooks/fastapi/src/chapter5).
# ```

# This notebook focuses on one of the most interesting parts of FastAPI: **dependency
# injections**. This is a powerful and readable approach to reuse logic across
# a project. An authentication system, a query parameters' validator, or a rate-limiter are typical use cases for dependencies. A dependency injection can even call another one recursively, allowing you to build high-level blocks from basic features. 

# ## What is dependency injection?

# A **dependency** is a way to wrap some logic that will retrieve some subvalues or instantiate subobjects, make something with them, and finally return a value that will be injected into the endpoint calling it. The responsibility of developers is then to only provide a declaration of how an object should be created, and let the system resolve all the dependency chains and create the actual objects at runtime.

# Using dependency injections allow for cleaner code by separation of concern between the logic of the endpoint and more generic logic (which can be used in several other endpoints in the project, at different scopes as we shall see), and more readable code since you know what the endpoint expects in the request data without reading the function's code. Finally, in the case of FastAPI, it's used to generate the OpenAPI schema so that the automatic documentation can clearly show which parameters are expected for this endpoint.

# ```python
# # 01_dependency_injection.py
# @app.get('/')
# async def header(user_agent: str = Header(...)):
#     return {"user_agent": user_agent}
# ```

# The `header` endpoint returns the header as a JSON response:

# In[22]:


get_ipython().system('http -v :8000')


# Internally, the `Header` function has some logic to automatically get the request object, check for the required header, return its value, or raise an error if it's not present. From our perspective, we don't know how it handled the required objects for this operation: we just assigned it to a variable `user_agent` with type `str`. The code for the endpoint reflects the endpoint logic: get the request header. If used well, dependency injections can make the code effortlessly readable and overwhelmingly powerful.

# ## Creating and using a function dependency

# A dependency can be defined using any **callable object** by wrapping it with the `Depends` constructor. In this section, we define dependencies using functions.

# ```python
# # 02_function_dependency.py
# from fastapi import FastAPI, Depends
# from typing import Tuple
# app = FastAPI()
# 
# async def pagination(skip: int = 0, limit: int = 10) -> Tuple[int, int]:
#     return (skip, limit)
# 
# @app.get("/items")
# async def list_items(p: Tuple[int, int] = Depends(pagination)):
#     skip, limit = p
#     return {"skip": skip, "limit": limit}
# ```

# FastAPI will recursively handle the arguments on the dependency and match them with the request data, such as query parameters or headers, if needed. The path function `list_items` uses the pagination dependency via the `Depends` function. The `Depends` function takes a callable and executes it when the endpoint is called. The subdependencies are automatically discovered
# and executed. This allows us to make the following query:

# In[24]:


get_ipython().system('http :8000/items skip==3 limit==5')


# Thus, `pagination` can be used in multiple places, i.e. we don't have to repeat the same query parameters in all functions! If we decide to change the API, we only have to make changes in one place. Moreover, we can add further validation on the query parameters, e.g. `limit: int = Query(..., ge=0, le=1000)`. The code on our path operation functions doesn't have to change: we have a clear separation of concern between the logic of the endpoint and the more generic logic for
# pagination.

# ### Get an object or raise a 404 error

# When dealing with databases, checking whether an object exists in the database is a common pattern. This is a perfect use case for a dependency.

# ```python
# # 03_function_dependency.py
# ...
# 
# posts = {
#     1: Post(id=1, title="Post #1"),
#     2: Post(id=2, title="Post #2"),
# }
# 
# async def get_post_or_404(id: int) -> Post:
#     try:
#         return db.posts[id]
#     except KeyError:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
# 
# 
# @app.get("/posts/{id}")
# async def get_post(post: Post = Depends(get_post_or_404)):
#     return post
# ```

# The path parameter `id` is passed to the dependency which results in a `post` object being injected into the endpoint.

# In[32]:


get_ipython().system('http :8000/posts/2')


# For a missing post we get `404 Not Found`:

# In[31]:


get_ipython().system('http :8000/posts/3')


# In addition to getting posts from the DB, we can now use this for when we have to update and delete posts. The key takeaway in this example is that you can **raise errors** in your dependencies. This can be useful when checking preconditions before your endpoint logic is executed, e.g. checking for existence of keys before retrieval.

# ## Creating and using a parameterized dependency with a class 

# In the previous section, we defined dependencies as regular functions, which works well
# in most cases. Still, you may need to set some parameters on a dependency to fine tune
# its behavior. Since the arguments of the function are set by the dependency injection
# system, we can't add an argument to the function.
# In the pagination example, we added some logic to cap the limit value at 100.
# If we wanted to set this maximum limit dynamically, how would we do that?

# ### Setting dependency attributes

# The solution is to create a class that will be used as a dependency with an `__init__` function to set attributes, and a `__call__` method to define the dependency logic. As stated above, that is all that `Depends` requires for a dependency: being a callable.

# ```python
# # 04_class_dependency.py
# class Pagination:
#     def __init__(self, maximum_limit: int = 100):
#         self.maximum_limit = maximum_limit
#     
#     async def __call__(self,
#         skip: int = Query(0, ge=0), limit: int = Query(10, ge=0),
#     ) -> Tuple[int, int]:
#         capped_limit = min(self.maximum_limit, limit)
#         return (skip, capped_limit)
# ```

# The logic of the `__call__` function is the same logic as the `pagination` function defined above. The only difference here is that we have `maximum_limit` as a class attribute. In the example below, we hardcoded the value 50, but we could very well pull it from a configuration file
# or an environment variable.

# ```python
# pagination = Pagination(maximum_limit=50)
# 
# @app.get("/items")
# async def list_items(p: Tuple[int, int] = Depends(pagination)):
#     skip, limit = p
#     return {"skip": skip, "limit": limit}
# ```

# Testing:

# In[37]:


get_ipython().system('http :8000/items skip==3 limit==600')


# ### Maintaining values in memory

# The other advantage of a class dependency is that it can maintain local values in memory.
# This property can be very useful if we have to make some heavy initialization logic, such
# as loading a machine learning model.
# Then, the callable part just has to call the loaded model to make the prediction, which
# should be quite fast, i.e. execute `__call__` without having to re-initialize the dependency. ❗
# 
# This setup would be useful for applications such as recommender systems which serves hundreds of requests per second. As such, the recommender model must be loaded at the start of the server, instead of at every prediction.

# ### Using class methods as dependencies

# As we said, `Depends` only needs a callable object. So you can pass class methods which can be very useful if you have common parameters or methods that makes sense to be put together in one class. For example, you could have one pretrained machine learning model made with Scikit-learn. Before applying the inference, you may want to apply different preprocessing steps depending on the specifics of the input data.

# ## Using dependencies at a path, router, and global level

# As we said, dependencies are the recommended way to create building blocks in a FastAPI
# project, allowing you to reuse logic across endpoints while maintaining maximum code
# readability. Until now, we've applied them on a single endpoint, but couldn't we expand
# this approach to a whole router? Or even a whole FastAPI application? Actually, we can!

# The main motivation for this is to be able to apply some global request validation or
# perform side logic on several routes without the need to add the dependency on each
# endpoint. Typically, an authentication method or a rate-limiter could be very good
# candidates for this use case. Consider the following dependency which raises a `403 Forbidden` whenever a request does not have the secret header. (Please note that this approach is only for the sake of the example. There are better ways to secure your API.)

# ```python
# def secret_header(secret_header: Optional[str] = Header(None)) -> None:
#     if (not secret_header) or (secret_header != "SECRET_VALUE"):
#         raise HTTPException(status.HTTP_403_FORBIDDEN)
# ```

# ### Using a dependency on a path decorator

# Note that we're not returning a value in the `secret_header` function. So we can't really assign its output to a parameter. FastAPI allows dependencies to be specified in the path decorator which we can do instead. (See also {numref}`dependency-tree`.)

# ```python
# @app.get("/protected-route", dependencies=[Depends(secret_header)]) # Add > 1 deps
# async def protected_route():
#     return {"hello": "world"}
# ```

# Recall the syntax for assigning a header:

# In[59]:


get_ipython().system('http -v :8000/protected-route "Secret-Header: SECRET_VALUE"')


# Trying to access the endpoint without the secret header should result in a 403:

# In[63]:


get_ipython().system('http :8000/protected-route')


# ### Using a dependency on a whole router

# If we want to protect a whole router, we can use:
# 
# ```python
# router = APIRouter(dependencies=[Depends(secret_header)])
# app.include_router(router, prefix="/router")
# ```
# 
# or 
# 
# ```python
# router = APIRouter()
# app.include_router(router, prefix="/router", dependencies=[Depends(secret_header)])
# ```

# If a router cannot run without the dependency, then the first approach is more readable. Otherwise, use the second approach. See {numref}`dependency-tree` below. In any case, the two approaches are equivalent as shown below.

# In[ ]:


get_ipython().system('http :8000/router/v1/route-1 "Secret-Header: SECRET_VALUE"')


# In[ ]:


get_ipython().system('http :8000/router/v2/route-1 "Secret-Header: SECRET_VALUE"')


# Trying to access a router endpoint without the secret value:

# In[ ]:


get_ipython().system('http :8000/router/v2/route-1')


# ### Use a dependency on a whole application 

# Similar to the router syntax, we pass the list of dependencies on the `app` constructor to make a **global dependency injection**.

# ```python
# app = FastAPI(dependencies=[Depends(secret_header)])
# ```

# The following tree is a guide to determine what level should you inject your dependencies:

# ```{figure} ../../img/dependency-tree.png
# ---
# name: dependency-tree
# ---
# At which level should I inject my dependency?
# ```

# 
