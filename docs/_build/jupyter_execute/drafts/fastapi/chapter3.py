#!/usr/bin/env python
# coding: utf-8

# # Developing a RESTful API

# ```{admonition} Attribution
# This notebook follows Chapter 3: *Developing a RESTful API with FastAPI* of {cite}`Voron2021`. Source files for running the background local servers can be found [here](https://github.com/particle1331/machine-learning/tree/master/docs/notebooks/fastapi/src/chapter3).
# ```

# This notebook covers the basics of creating API endpoints in 
# FastAPI. We will go through simple and focused examples that will demonstrate the different features of FastAPI. Each example will lead to a working API endpoint that can be tested locally using HTTPie.

# ## Hello, World!

# Let us quickly create a simple endpoint which has a GET method.
# 
# ```python
# # chapter3/hello_world.py
# from fastapi import FastAPI
# 
# app = FastAPI()
# 
# @app.get("/")
# async def hello_world():
#     return {"hello": "world"}
# ```

# The path function `hello_world` contains our route logic for the path `/` specified in the decorator. The decorator also specifies what HTTP method this function implements. The return value is automatically handled by FastAPI to produce a proper HTTP response with a JSON payload.

# Here `app` is the main application object that will wire all of the API routes. We will 
# start the server in the terminal as follows: 
# 
# ```
# $ uvicorn hello_world:app --reload
# INFO:     Started server process [14121]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# ```
# 
# Then, we perform the following GET request.

# In[22]:


import requests
root = "http://127.0.0.1:8000"
path = "/"

response = requests.get(root + path)
response.json()


# ## HTTPie

# Before delving into the details of building REST APIs, we need to have a tool for making HTTP requests. We will be using **HTTPie**, a command-line tool aimed at making HTTP
# requests with an intuitive syntax, JSON support, and syntax highlighting.

# > HTTPie (pronounced *aitch*-*tee-tee-pie*) is a command-line HTTP client. Its goal is to make CLI interaction with web services as human-friendly as possible. HTTPie is designed for testing, debugging, and generally interacting with APIs & HTTP servers. The `http` & `https` commands allow for creating and sending arbitrary HTTP requests. They use simple and natural syntax and provide formatted and colorized output.

# The general form of an HTTPie request is:

# ```
# http [flags] [METHOD] URL [ITEM [ITEM]]
# ```

# For our local server, we can call:

# In[24]:


get_ipython().system('http -v GET http://127.0.0.1:8000/')


# The flag `-v` or `--verbose` is used here to print both the request and response. HTTPie also provides an `https` executable for dealing with URLs with `https://`. 

# ### Optional GET and POST

# The `METHOD` argument is optional, and when you don’t specify it, HTTPie defaults to:
# 
# - `GET` for requests without body
# - `POST` for requests with body

# For example, the following is a GET request. 

# In[21]:


# !http -v GET http://127.0.0.1:8000/
get_ipython().system('http -v http://127.0.0.1:8000/')


# On the other hand, the following request has data, so that the following defaults to a POST request. We add the function below which has a POST method to our main app. 

# ```python
# # chapter3/hello_world.py
# 
# @app.post("/")
# async def message(msg: str = Body(..., embed=True)):
#     return {"message": msg}
# ```

# In[25]:


get_ipython().system('http -v http://127.0.0.1:8000/ msg="Hello"')


# ### Querystring parameters 

# HTTPie provides `param==value` syntax for appending URL querystring parameters. With that, you don’t have to worry about escaping the `&` separators for your shell. Thus, the following are then equivalent:

# In[ ]:


get_ipython().system('http :8000/ p==foo q==bar')
get_ipython().system('http ":8000/?p=foo&q=bar"')


# ### URL shortcuts for localhost

# Shorthand for `localhost` is supported. For example, `:8000` would expand to `http://localhost:8000`. If the port is omitted, then port 80 is assumed.

# In[11]:


get_ipython().system('http :8000/')


# ## Automatic documentation

# One of the most beloved features of FastAPI is the automatic interactive documentation.
# If you open `http://localhost:8000/docs` in your browser, you should get a web
# interface that looks similar to the following screenshot:

# ```{figure} ../../img/fastapi-docs.png
# ---
# width: 40em
# name: fastapi-docs
# ---
# 
# ```

# FastAPI automatically lists all defined endpoints and provide documentation about the expected inputs and outputs. You can even try each endpoint directly in this web interface.

# ## Handling request parameters

# The main goal of a REST API is to provide a structured way in which to interact with data. As such, it's crucial for the end-user to send some information to tailor the response they need, such as the following: 
# 
#    - path parameters
#    - query parameters
#    - body payloads
#    - headers

# To handle them, usually, web frameworks ask you to manipulate a request object to retrieve 
#    the parts you are interested in and manually apply validation. However, that's not necessary 
#    with FastAPI. Indeed, it allows you to define all of your parameters declaratively. Then, 
#    it'll automatically retrieve them in the request and apply validations based on the type 
#    hints as we will see below.

# ### Path parameters

# We can have dynamic parameters in our paths which can then be passed to the path function. For example:
# 
# ```python
# # chapter3/path_parameters.py
# 
# @app.get("/v1/users/{id}")
# async def get_user(id: int):
#     return {"id": id}
# ```

# Then, we can make the following request for `id=123` (or for any other integer):

# In[29]:


get_ipython().system('http :8000/v1/users/123')


# Notice the **type hint** in the path parameter `id`. If we pass a string into `id`, we get a response with a 422 status! Since this cannot be converted as a valid integer, the validation fails and outputs an error. All we need to do to trigger this validation is to type hint our parameter! Very cool.

# #### Validation logic for path parameters
# 
# **Enumeration**. In the example below, `type` is a categorical parameter with two accepted values. We inherit from the `str` type and `Enum` class to facilitate the intended typing. We simply list the property name and its actual string value. 

# ```python
# # chapter3/path_parameters.py
# 
# class UserType(str, Enum):
#     STANDARD = "standard"
#     ADMIN = "admin"
# 
# 
# @app.get("/v2/users/{type}/{id}")
# async def get_user(id: int, type: UserType):
#     return {"id": id, "type": type}
# ```
# 

# Note that the actual string value is what is passed in the `type` parameter (not the property name). If we pass a value that is not in the enumeration, we get an error.

# In[30]:


get_ipython().system('http :8000/v2/users/admin/3')


# In[31]:


get_ipython().system('http :8000/v2/users/dog/3')


# **Integer bounds**. For integers we can use the `Path` object from the `fastapi` library. In the example below, we set a lower bound to `id` so that it only takes positive values. 

# ```python
# # chapter3/path_parameters.py
# from fastapi import FastAPI, Path
# app = FastAPI()
# 
# @app.get("/v3/users/{id}")
# async def get_user(id: int = Path(..., ge=1)):
#     return {"id": id, "type": type}
# ```

# i.e. `id >= 1`. Other possible arguments are `gt`, `lt`, `le`, etc. The `Path` function 
# requires a first argument which becomes the default argument, using `...` indicates that 
# we don't want to set a default argument. 
# 

# In[32]:


get_ipython().system('http :8000/v3/users/3')


# In[33]:


get_ipython().system('http :8000/v3/users/-1')


# **Strings and regex**. We can bound string length using `min_length` and `max_length`. More generally, we can parse a string with a regular expressions in the `regex` argument.
# 
# ```python
# # chapter3/path_parameters.py
# 
# @app.get("/username/{username}")
# async def get_username(username: str = Path(..., min_length=1, max_length=20)):
#     return {"username": username}
# 
# @app.get("/license-plates/{license}")
# async def get_license_plate(license: str = Path(..., regex=r"^\w{2}-\d{3}-\w{2}$")):
#     return {"license": license}
# ```
# 
# 

# In[36]:


get_ipython().system('http :8000/license-plates/AB-123-CD')


# In[38]:


get_ipython().system('http :8000/username/abcdefghijklmnopqrst # 20 characters')


# ```{admonition} Parameter metadata
# Data validation is not the only option accepted by the parameter function `Path`. 
# You can also set options such as `title`, `description`, and `deprecated`. These will add information about the parameter 
# in the automatic documentation.

# ### Query parameters

# Query parameters are a common way to add some dynamic parameters to a 
# URL. You find them at the end of the URL in the following form: `?param1=foo&param2=bar`. 
# In a REST API, they are commonly used on read endpoints to apply pagination, a filter, a 
# sorting order, or selecting fields.

#  By default, arguments of path functions that are not path parameters are interpreted by 
#     FastAPI as query parameters (i.e. without having to use the `Query` function defined 
#     below).

# ```python
# # chapter3/query_parameters.py
# 
# @app.get("/users/{type}/{id}")
# async def user(
#     type: UserType, 
#     id: int = Path(..., ge=1), 
#     page: int = 1,
#     size: int = 1):
#     return {
#         "type": type,
#         "id": id,
#         "page": page, 
#         "size": size,
#     }
# ```

# In[40]:


get_ipython().system('http :8000/users/admin/1 page==0 size==3')


# These parameters are optional since we defined a default value. If there is no default
#     value, then query parameter has to be supplied. If we want to validate query parameters, 
#     we use the `Query` function from `fastapi`. This works in the same way as the `Path` 
#     function. We can add metadata as mentioned in the above tip.

# ### The request body

# The body is the part of the HTTP request that contains is used to send 
#     and receive data via the REST API representing documents, files, or form submissions. In a REST API, it's usually encoded in JSON and used to create structured objects in a 
#     database. 
#     
#     
# For the simplest cases, retrieving data from the body works exactly like query 
#     parameters. The only difference is that you always have to use the `Body` function; 
#     otherwise, FastAPI will look for it inside the query parameters by default. 

# 
# ```python
# # chapter3/request_body.py
# 
# @app.post("/users")
# async def user(
#     name: str = Body(...), 
#     age:  int = Body(...),
# ):
#     return {
#         "name": name, 
#         "age":  age,
#     }
# ```

# To send a request body, we use the following syntax in HTTPie. Note that HTTPie automatically detects that this is a POST method since a body is present. We see that a JSON payload was sent with string parameters. Note that the value of `age` in the response is automatically converted to `int` consistent with the typing.

# In[44]:


get_ipython().system('http -v :8000/v1/users name=Ronnie age=-99')


# Advanced validation and metadata through the `Body` function works in the same way as `Path` and `Query`. 

# ```{warning}
# For some reason, having a single `Body` parameter results in unexpected behavior. Indeed this is documented in [Issue #1097](https://github.com/tiangolo/fastapi/issues/1097). The solution is to set `embed=True` in the `Body` function for the single argument. Or defining a Pydantic model for this parameter as we will show below. This is a little inconsistent but done for the sake of backward compatibility. 
# ```

# #### Pydantic models

# Defining payload validations like this has some major drawbacks. First, it's quite verbose and makes the path function prototype huge, especially for bigger models with many fields. Second, usually, you'll need to reuse the data structure on other endpoints or in other parts of your application.
# 
# We'd like to define the data model in one place, so that updating the model updates all other places in the code where the model is used. For this we use **Pydantic models** which allows us to automatically type hint fields. We can use the `Field` function to validate Pydantic model fields. This offers the same options as the usual validation functions, e.g. `name: str = Field(..., max_length=20)`.

# ```python
# # chapter3/request_body.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# app = FastAPI()
# 
# class User(BaseModel):
#     name: str
#     age: int
# 
# @app.post("/v2/users")
# async def user(user: User):
#     return user
# ```

# In[46]:


get_ipython().system('http :8000/v2/users name=Ronnie age=-99')


# Note that FastAPI automatically understands that the user data can be found in the 
#     request payload and each field automatically validated based on the Pydantic model. 
#     Furthermore, the Pydantic object is simply returned and FastAPI is smart enough to convert it 
#     automatically into JSON to produce the HTTP response.

# We can extend this to multiple objects, as well as singular body values with the `Body` 
#     function. This is useful if you wish to have a single property that's not part of any 
#     model.

# ```python
# # chapter3/request_body.py
# 
# class User(BaseModel):
#     name: str
#     age: int
# 
# class Company(BaseModel):
#     name: str
# 
# @app.post("/v3/users")
# async def create_user(user: User, company: Company, priority: int = Body(..., ge=1, le=3)):
#     return {"user": user, "company": company, "priority": priority}
# ```

# This expects a properly formatted complex JSON structure as payload that can be passed using `<<<`.

# In[47]:


get_ipython().system('http :8000/v3/users <<< \'{ "user": {"name": "Ronnie", "age": -99 }, "company": { "name": "Alliance" }, "priority": 1 }\'')


# ### Form data

# We need to install `python-multipart` to handle form data.

# ```python
# # chapter3/form_data.py
# from fastapi import FastAPI, Form
# app = FastAPI()
# 
# @app.post("/users")
# async def create_user(name: str = Form(...), age: int = Form(...)):
#     return {"name": name, "age": age}
# ```

# The only difference is we use `Form` instead of `Body`. Validation and metadata options we saw for `Path`, `Query`, and `Body` are likewise available. Note that FastAPI does not allow Pydantic models to validate form data. 
# 
# We use the `--form` option enforces the data to be form-encoded:

# In[48]:


get_ipython().system('http -v --form :8000/users name=Ronnie age=96')


# Observe that `Content-Type: application/x-www-form-urlencoded; charset=utf-8` and the body data representation have changed in the request. Note also that the response for form data is still provided in JSON which is the default response by FastAPI no matter the form of input data. 

# ### File uploads

# To handle files, the approach is still the same: we define an argument for the 
# path operation function, `file`, we add a type of hint, `bytes`, and then we use the `File` 
# function as a default value for this argument. By doing this, FastAPI understands that it 
# will have to retrieve raw data in a part of the body named file and return it as bytes.

# ```python
# # chapter3/file_uploads.py
# from fastapi import FastAPI, File
# app = FastAPI()
# 
# @app.post("/v1/files")
# async def upload_file(file: bytes = File(...)):
#     return {"file_size": len(file)}
# ```

# In[50]:


get_ipython().system('http --form :8000/v1/files file@./assets/dog.jpeg')


# One drawback to this approach is that the uploaded file is entirely stored in the server's memory while the function processes the request. This will likely run into issues for larger 
#     files. Besides, manipulating a bytes object is not always convenient for file handling. To fix this problem, FastAPI provides an `UploadFile` class. This class will store the data 
#     in memory up to a certain threshold and, after this, will automatically store it on disk 
#     in a temporary location. 

# The exposed object instance exposes useful metadata, such as the 
#     content type, and a **file-like** interface. This means that you can manipulate it as a 
#     regular file in Python and that you can feed it to any function that expects a file. For example, `contents = await myfile.read()`.

# ```python
# # chapter3/file_uploads.py
# 
# @app.post("/v2/files")
# async def upload_file(file: UploadFile = File(...)):
#     return {
#         "file_name": file.filename, 
#         "content_type": file.content_type
#     }
# ```

# In[51]:


get_ipython().system('http --form :8000/v2/files file@./assets/dog.jpeg')


# Since `UploadFile` does not validate for specific file types, the content type is especially useful for validating the type of the uploaded file.

# **Multiple files.** To upload multiple files, we can use the `List` type hint.

# ```python
# # chapter3/file_uploads.py
# 
# @app.post("/v3/files")
# async def upload_multiple_files(files: List[UploadFile] = 
# File(...)):
#     return [
#         {
#             "file_name": file.filename,
#             "content_type": file.content_type
#         }
#         for file in files
#     ]
# ```

# Uploading multiple files via HTTPie:

# In[53]:


get_ipython().system('http --form :8000/v3/files files@./assets/dog.jpeg files@./assets/dog.jpeg')


# ### Headers and cookies

# Besides the URL and the body, another major part of the HTTP request 
#     are the **headers**. They contain all sorts of metadata that can be useful when handling 
#     requests. Another is **cookies** which allows servers to store stateful information. Headers and cookies can be very useful tools in which to implement some authentication 
# features. 

# ```python
# # chapter3/headers_cookies.py
# 
# @app.get("/")
# async def get_header(hello_world: str = Header(...), cookie: Optional[str] = Cookie(None)):
#     return {"hello_world": hello_world, "cookie": cookie}
# ```

# In[57]:


get_ipython().system('http -v :8000/ "Hello-World: Hi"')


# FastAPI automatically parses the header name to lowercase and snake case to get its corresponding variable in the function body. Here `"Hello-World"` to `hello_world`. 
# 
# One very special case of header is cookies. FastAPI provides another parameter function 
#     that automatically parses cookies for you. Here we didn't pass anything so its value is `null`. 

# ### The request object

# Sometimes, you might find that you need to access a raw request object 
#     with all of the data associated with it. Simply declare an argument on 
#     your path operation function type hinted with the `Request` class from the FastAPI library.

# ```python
# # chapter3/request_object.py
# from fastapi import FastAPI, Request
# app = FastAPI()
# 
# @app.get("/random-path")
# async def get_request_object(request: Request):
#     return {"path": request.url.path}
# ```

# In[58]:


get_ipython().system('http :8000/random-path')


# ## Customizing the response

# Most of the time, you'll want to customize this response a 
#     bit further; for instance, by changing the status code, raising validation errors, and 
#     setting cookies. We will discuss the different ways FastAPI does this, from the simplest case to the most advanced one.

# ### Decorator parameters

# In order to 
# create a new endpoint, you had to put a decorator on top of the path operation function. 
# This decorator accepts a lot of options, including ones to customize the response.

# #### Status code

# This is the most obvious. By default FastAPI sets `200 OK` when 
#     everything goes well in executing the path operation function. Sometimes, it might be 
#     useful to change this status. 

# ```python
# # chapter3/path_success_status_code.py
# from fastapi import FastAPI, status
# from pydantic import BaseModel
# app = FastAPI()
# 
# class Post(BaseModel):
#     title: str
# 
# # Dummy database
# posts = {}
# 
# @app.post("/posts", status_code=status.HTTP_201_CREATED)
# async def create_post(post: Post):
#     posts[len(posts) + 1] = post
#     return post
# 
# @app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_post(id: int):
#     posts.pop(id, None)
#     return None
# ```

#     
# For example, it's good practice in a REST API to return 
#     a `201 Created` status when the execution of the endpoint ends up in the creation of 
#     a new object. (We can test whether the created posts persist in memory using print statements. Indeed they do, as expected.)

# In[63]:


get_ipython().system('http :8000/posts title="Hello!"')


# In[64]:


get_ipython().system('http -v DELETE :8000/posts/1')


# **Remark**. This runs into an error discussed in [#2253](https://github.com/tiangolo/fastapi/issues/2253). Returning `None` has content-length 4 which seems to be the root cause of the error. A fix seems to be to return a response model with status 204. This has content-length 0.

# ```{warning}
# It's important to understand that this option to override the status code is 
#     only useful when everything goes well. If your input data was invalid, you would still get 
#     a 422 status error response.
# ```

# #### The response model

# The main use case in FastAPI is to directly return a pydantic model 
#     that automatically gets turned into properly formatted JSON. Quite often, you'll 
#     find that there are some differences between the input data, the data you store in your 
#     database, and the data you want to show to the end user. The **response model** allows us to specify the pydantic model for the response.

# Suppose we have a Pydantic model `Post` which which has
#     fields `title` and `nb_views`. If we want to hide `nb_views` in a response, we
#     can create a new Pydantic model `PublicPost` that inherits from `Post` which does not include `nb_views` in its fields. Then, we set `response_model=PublicPost` in the
#     path decorator.

# ```python
# # chapter3/path_response_model.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# app = FastAPI()
# 
# class Post(BaseModel):
#     title: str
#     nb_views: int
# 
# class PublicPost(BaseModel):
#     title: str
# 
# # Dummy database. Database models.
# posts = {
#     1: Post(title="Post #1", nb_views=100),
# }
# 
# @app.get("/posts/{id}", response_model=PublicPost)
# async def get_post(id: int):
#     return posts[id] # Public-facing model
# ```

# Note that `posts[id]` is an instance of the pydantic model `Post` which has `nb_views`. However, the resulting JSON only shows title in the public post and not the number of views. 
# 
# 
# The good thing is that this option is also considered by the interactive documentation, 
# which will show the correct output schema to the end user &mdash; hiding `nb_views`.

# In[66]:


get_ipython().system('http :8000/posts/1 # does not show nb_views')


# ### Injecting the response object

# The body and status code are not the only interesting parts of an HTTP response. 
# Sometimes, it might be useful to return some custom headers or set cookies. This can be 
# done dynamically using FastAPI directly within the path operation logic by 
# **injecting** the `Response` object as an argument of the path operation function.

# #### Setting headers and cookies

# Sometimes, it might be useful to return some custom headers or set 
#     cookies. The good thing about this approach is that it's within your 
# path operation logic. That means you can dynamically set headers depending on what's 
# happening in your business logic.

# ```python
# # chapter3/response_object.py
# from fastapi import FastAPI, Response, Request
# app = FastAPI()
# 
# @app.get("/custom-header")
# async def custom_header(response: Response):
#     response.headers["Custom-Header"] = "Custom-Header-Value"
#     return {"hello": "world"}
# 
# @app.get("/custom-cookie")
# async def custom_cookie(response: Response):
#     response.set_cookie("cookie-name", "cookie-value", max_age=86400) # -> name value pair
#     return {"hello": "world"}
# ```

# In[67]:


get_ipython().system('http :8000/custom-header')


# Note you don't have to return `response` (though this is the natural thing to do). You can 
#     still return JSON-encodable data and FastAPI will take care of forming a proper response, 
#     including the headers you've set. Therefore, the `response_model` and `status_code` options 
#     we discussed are still honored [^ref].
# 
# [^ref]: Assuming these are not set in the response object! The attributes of the response object **overrides** the corresponding parameters in the decorator. See warning at the end of the *Building a custom response* section. 

# Cookies can also be particularly useful when you want to maintain the 
#     user's state within the browser between each of their visits. The `Response` object above exposes 
#     a convenient `set_cookie` method. We will see the ff. header added to the response:
#     `set-cookie: cookie-name=cookie-value; Max-Age=86400; Path=/; SameSite=lax`

# In[68]:


get_ipython().system('http :8000/custom-cookie')


# #### Setting status code dynamically

# Recall we set the `status_code` parameter in the
#     path function decorator. The drawback to this approach is that it'll always be the same 
#     no matter what's happening inside. This can be set dynamically in the `response` object. Again, this overrides the corresponding parameter in the decorator.

# ```python
# # chapter3/response_object_dynamic_status_code.py
# 
# @app.put("/posts/{id}")
# async def update_or_create_post(id: int, post: Post, response: Response):
#     if id not in posts.keys():
#         response.status_code = status.HTTP_201_CREATED
#     posts[id] = post
# ```

# Creating a new post should result in status `201 Created`.

# In[71]:


get_ipython().system('http PUT :8000/posts/2 title="Post #2" nb_views=1')


# Updating an old post results in vanilla `200 OK`:

# In[72]:


get_ipython().system('http PUT :8000/posts/1 title="Post #1" nb_views=2')


# Note: don't use this pattern to set error status code, e.g. `404 Not Found`. See next
#     section. Also, dynamic status codes are not detected by the automatic documentation.

# ### Raising HTTP errors

# Errors can happen for a lot of reasons: wrong parameters, invalid 
#     payloads, or objects that don't exist anymore. That's why it's critical to detect them and 
#     raise a clear and unambiguous error message to the end user so that they can correct their 
#     mistake. Two places to return error: payload and status code.

# ```python
# # chapter3/raise_errors.py
# 
# @app.post("/password")
# async def check_password(password: str = Body(...), password_confirm: str = Body(...)):
#     if password != password_confirm:
#         raise HTTPException(
#             status.HTTP_400_BAD_REQUEST,
#             detail="Passwords don't match.",
#         )
#     return {"message": "Passwords match."}
# ```

# In[73]:


get_ipython().system('http :8000/password password=a123 password_confirm=A123')


# Here, we do get a 400 status code and our error message has been wrapped nicely in a 
#     JSON object with the detail key. This is how FastAPI handles errors by default. We can
#     make a more detailed description by returning a dictionary for the `detail` parameter:
# 
# ```PYTHON
# detail={
#     "message": "Passwords don't match.",
#     "hints": [
#         "Check the caps lock on your keyboard",
#         "Try to make the password visible by clicking on the eye icon to check your typing",
#         ],
# },
# ```

# ### Building a custom response

# So far, all of the methods you have seen should cover the majority of cases you'll 
# encounter during the development of an API. Sometimes, however, you'll have scenarios 
# where you'll need to build a complete HTTP response yourself. This is the subject of the 
# next section

# Under the hood, FastAPI uses a subclass of `Response`, called 
#     `JSONResponse`. Quite predictably, this response class takes care of serializing some 
#     data to JSON and adding the correct `Content-Type` header.

# - `HTMLResponse`: This can be used to return an HTML response.
# - `PlainTextResponse`: This can be used to return raw text.
# - `RedirectResponse`: This can be used to make a redirection.
# - `StreamingResponse`: This can be used to stream a flow of bytes.
# - `FileResponse`: This can be used to automatically build a proper file response given 
# the path of a file on the local disk. 

# You have two ways of using them: either setting the `response_class` argument on the 
# path decorator or directly returning a response instance.
# 

# #### Using the response_class argument

# ```python
# @app.get("/html", response_class=HTMLResponse)
# async def get_html():
#     return """
#     <html>
#         <head>
#             <title>Hello world!</title>
#         </head>
#         <body>
#             <h1>Hello world!</h1>
#         </body>
#     </html>
#     """
# 
# @app.get("/text", response_class=PlainTextResponse)
# async def text():
#     return "Hello world!" 
# ```

# #### Making a redirection

# For the rest of responses class, we have to build the response 
#     instance. `RedirectResponse` is a class that helps you build an HTTP redirection, which 
#     simply is an HTTP response with a Location header pointing to the new URL and a status 
#     code in the `3XX` range. It simply expects the URL you wish to redirect to as the first 
#     argument:
# 
# ```python
# @app.get("/redirect")
# async def redirect():
#     return RedirectResponse("/new-url") # 307 Temporary Redirect
# ```

# #### Serving a file

# `FileResponse` will be useful for downloading files. This response class will automatically take care of 
#     opening the file on disk and streaming the bytes along with the proper HTTP headers. We may need to install `aiofiles` as a dependency.

# In[2]:


get_ipython().system('pip install aiofiles')


# ```python
# # chapter3/file_response.py
# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from pathlib import Path
# app = FastAPI()
# 
# @app.get("/dog")
# async def get_dog():
#     root_directory = Path(__file__).absolute().parents[2]
#     img_path = root_directory / "assets" / "dog.jpeg"
#     return FileResponse(img_path)
# ```

# In[75]:


get_ipython().system('http :8000/dog')
# --download to download image


# In the browser, the image can be displayed:

# ```{figure} ../../img/dog.png
# ---
# width: 40em
# name: dog
# ---
# 
# ```

# #### Custom responses

# Finally, if you really have a case that's not covered by the provided 
#     classes, you always have the option to use the `Response` class to build exactly what you 
#     need. With this class, you can set everything, including the body content and the headers.
#     The following example shows you how to return an XML response:
# 
# ```python
# @app.get("/xml")
# async def get_xml():
#     content = """
#         <?xml version="1.0" encoding="UTF-8"?>
#         <Hello>World</Hello>
#     """
#     return Response(content=content, media_type="application/xml")
# ```
# 

# ```{warning}
# Bear in mind that when you directly return a `Response` class (or one of 
# its subclasses), the parameters you set on the decorator won't have any effect. They are completely overridden by the 
# `Response` object you return. If you need to customize the status code or the headers, then use the `status_code` and 
# `headers` arguments when instantiating the response object.
# ```

# ## Structuring a bigger project with multiple routers

# When building a real-world web 
# application, you're likely to have lot of code and logic: data models, API endpoints, and 
# services. Of course, all of those can't live in a single file; we have to structure the 
# project so that it's easy to maintain and evolve.
# 
# ```
# .
# └── chapter3_project/
#     ├── models/
#     │ ├── __init__.py
#     │ ├── post.py
#     │ └── user.py
#     ├── routers/
#     │ ├── __init__.py
#     │ ├── posts.py
#     │ └── users.py
#     ├── __init__.py
#     ├── app.py
#     └── db.py
# ```

# Everything is the same, but you separate functionality into separate files each can be thought of as a subapplications, then connect everything together by using **routers**. 

# ```python
# # chapter3_project/routers/users.py
# from typing import List
# from fastapi import APIRouter, HTTPException, status
# from chapter3_project.models.user import User, UserCreate
# from chapter3_project.db import db
# router = APIRouter()
# 
# @router.get("/")
# async def all() -> List[User]:
#     return list(db.users.values())
# ```
# 
# As you can see here, instead of instantiating the FastAPI class, you instantiate the 
# APIRouter class. Then, you can use it exactly the same way to decorate your path 
# operation functions. Now, let's take a look at how to import this router and include it 
# within a FastAPI application:
#     
# ```python
# from fastapi import FastAPI
# from chapter3_project.routers.posts import router as posts_router
# from chapter3_project.routers.users import router as users_router
# 
# app = FastAPI()
# 
# app.include_router(posts_router, prefix="/posts", tags=["posts"])
# app.include_router(users_router, prefix="/users", tags=["users"])
# ```
# 
# Prefixes can also be used to provide versioned paths of your API, such as `/v1`. The 
# tags argument helps you to group endpoints in the interactive documentation for better 
# readability. By doing this, the posts and users endpoints will be clearly separated in 
# the documentation.
# 
# Once again, you can see that FastAPI is both powerful and very lightweight to use. The 
# good thing about routers is that you can even nest them, that include sub-routers in 
# routers that include other routers themselves. Therefore, you can have a quite complex 
# routing hierarchy with very low effort.
# 
# ```python
# router.include_router(subrouter, prefix="/subrouter-url", tags=["router"])
# ```
#     
# 

# 
