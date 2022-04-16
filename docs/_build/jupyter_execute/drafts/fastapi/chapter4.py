#!/usr/bin/env python
# coding: utf-8

# # Managing Pydantic Data Models

# ```{admonition} Attribution
# This notebook follows Chapter 4: *Managing Pydantic Data Models in FastAPI* of {cite}`Voron2021`. Source files for running the background local servers can be found [here](https://github.com/particle1331/machine-learning/tree/master/docs/notebooks/fastapi/src/chapter4).
# ```

# This chapter will cover in more detail the definition of a data model with Pydantic, the 
# underlying data validation library used by FastAPI.  We'll explain how to implement
# variations of the same model without repeating the same code again and again, thanks to
# class inheritance. Finally, we'll show how to implement custom data validation logic into
# Pydantic models.

# ## Defining models and their field types with Pydantic

# Pydantic is a powerful library for defining data models using Python classes and type 
# hints. This approach makes those classes completely compatible with static type checking.
# Moreover, since there are regular Python classes, we can use inheritance and also define our
# very own methods to add custom logic.

# ### Standard field types

# We can have quite quite complex field types &mdash; usual static field types (`int`, `str`, `float`), dynamic field types (`list[str]`, `date`), as well as fields which are themselves Pydantic models.

# In[1]:


from datetime import date
from enum import Enum
from pydantic import BaseModel


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"

class Address(BaseModel):
    street_address: str
    postal_code: str
    city: str
    country: str

class Person(BaseModel):
    id: int
    first_name: str
    last_name: str
    address: Address
    gender: Gender
    birthdate: date 
    interests: list[str]


# Let's instantiate one example:

# In[2]:


Person(
    id=3,
    first_name="Ron", 
    last_name="Medina", 
    address=Address(
        street_address="#1 Street",
        postal_code="333",
        city="City",
        country="Country"),
    gender=Gender.MALE,
    birthdate=date(2021, 7, 21), # YYYY-MM-DD
    interests=[],
)


# Observe that the Pydantic model performs automatic validation and type conversion:

# In[3]:


Person(
    id="3",
    first_name="Ron", 
    last_name="Medina", 
    gender="male",
    birthdate="2021-07-21",
    interests=[],
    address={
        "street_address": "#1 Street",
        "postal_code": "333",
        "city": "City",
        "country": "Country",
    }
)


# ### Optional fields and default values

# Be careful: don't assign default values that
#     are dynamic types such as datetimes. By doing so, the datetime instantiation will be 
#     evaluated *only once* when the model is imported. The effect of this is that all the 
#     objects you'll instantiate will then share the same value instead of having a fresh value. This is a known Python gotcha. 
# 

# In[4]:


import time 
from datetime import datetime
from typing import Optional

class Post(BaseModel):
    date_created: date = datetime.now()
    tag: Optional[str]


post1 = Post()
time.sleep(3)
post2 = Post()
print((post1.date_created - post2.date_created).total_seconds())


# Fortunately, Pydantic provides a `Field` 
# function that allows us to set some advanced options on our fields, including one to set 
# a factory for creating dynamic values.

# ### Field validation

# It turns out that the validation for request parameters come directly from Pydantic. The syntax is very similar to the one we saw for `Path`, `Query`, and `Body`. The `Field` function validates the data fields of a Pydantic model. For optional fields (`age`), validation only applies if a value is supplied. 

# In[5]:


from pydantic import BaseModel, Field, ValidationError

class Person(BaseModel):
    first_name: str = Field(..., min_length=3)
    last_name: str = Field(..., min_length=3)
    age: Optional[int] = Field(ge=0, le=120)


# Let's trigger an error with first name of length zero.

# In[6]:


try:
    p = Person(first_name="", last_name="Shorttail")
except ValidationError as e:
    print(e)


# #### Default factories for dynamic default values

# Recall gotcha with dynamic or mutable default values. Pydantic provides the `default_factory` argument on the `Field` function to cover this use case. This argument expects you to pass a function that will be 
# called during model instantiation. Thus, the resulting object will be evaluated at runtime 
# each time you create a new object. 

# In[7]:


class Post(BaseModel):
    date_created: date = Field(default_factory=lambda: datetime.now())
    tag: Optional[str]

post1 = Post()
time.sleep(3)
post2 = Post()
print((post2.date_created - post1.date_created).total_seconds())


# The factory function should have no arguments. Moreover, there is no need to set default values in the `Field` functions (which is reasonable).

# ### Validating email addresses and URLs with Pydantic types

# For this to work, you may need `email-validator` which can be installed using `pip`. 

# In[8]:


get_ipython().system('pip install email-validator')


# In[9]:


from pydantic import BaseModel, EmailStr, HttpUrl, ValidationError

class User(BaseModel):
    email: EmailStr
    website: HttpUrl


# In the following script, we use `ValidationError` as our exception class:

# In[10]:


try:
    User(email="user@email,com", website="https://www.example.com")
except ValidationError as e:
    print(str(e))


# In[11]:


try:
    User(email="user@email.com", website="https://www.example,com")
except ValidationError as e:
    print(str(e))


# When valid, we get the following parsing for the URL:

# In[12]:


User(email="jdoe@example.com", website="https://www.example.com")


# ## Creating model variations with class inheritance

# Recall in the previous notebook, we saw a case where we needed to 
# define two variations of a Pydantic model in order to split between (1) the data we want to 
# store in the backend and (2) the data we want to show to the user. We can another model which represents (3) the data needed to create an object. This is a common pattern 
# in FastAPI: you define one model for **creation**, one for the **response** and one for the **database entry**.

# In[13]:


class PostBase(BaseModel):
    title: str
    content: str

class PostCreate(PostBase):
    pass

class PostPublic(PostBase):
    id: int

class PostDB(PostBase):
    id: int
    nb_views: int = 0


# These models cover the three situations described above. `PostCreate` will be used for a POST endpoint to create a new post. We expect the user to give the title and the content, while the **identifier** `id` will be automatically determined by the database. `PostPublic` will be used when we retrieve the data of a post. We want its title and content, of course, but also its associated ID in the database. `PostDB` will carry all the data we wish to store in the database. Here, we also want
# to store the number of views, but we want to keep this secret to make our own statistics internally.

# We use class inheritance to adhere to the [DRY principle](https://en.wikipedia.org/wiki/Don't_repeat_yourself). Here the three models depend on a single base model. Note that we can also add methods on the base model which will be inherited by the derived models.

# ## Adding custom data validation with Pydantic

# In a real-world project, though,
# you'll probably need to add your own custom validation logic for your specific case.
# Pydantic allows this by defining **validators**, which are methods on the model that can be
# applied at a field level or an object level.

# ### Appying validation at a field level

# In[14]:


from datetime import date
from pydantic import BaseModel, validator

class Person(BaseModel):
    first_name: str
    last_name: str
    birthdate: date

    @validator("birthdate")
    def valid_birthdate(cls, v: date): # classmethod
        delta = date.today() - v
        age = delta.days / 365
        if age > 120:
            raise ValueError("You seem a bit too old!")
        return v


# Let's test with a person born in the 1800s.

# In[15]:


try:
    Person(first_name="John", last_name="Doe", birthdate="1800-01-01")
except ValidationError as e:
    print(e)


# In[16]:


person = Person(first_name="John", last_name="Doe", birthdate="1991-01-01")
print(person)


# Pydantic expects two things for this method. If the value is not valid according to your logic, you should raise a **`ValueError`** error with an explicit error message. Otherwise, you should return the value that will be assigned in the model. Notice that it doesn't need to be the same as the input value: you can very well change it
# to fit your needs.

# ### Applying validation at an object level

# It happens quite often that the validation of one field is dependent on another &mdash; for
# example, to check if a password confirmation matches the password or to enforce a field
# to be required in certain circumstances. To allow this kind of validation, we need to access
# the *whole* object data. For this, Pydantic provides the `root_validator` decorator,
# which is illustrated in the following code example:

# In[17]:


from pydantic import BaseModel, root_validator, EmailStr, ValidationError

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    password_confirmation: str

    @root_validator()
    def passwords_match(cls, values):
        password1 = values.get("password")
        password2 = values.get("password_confirmation")
        if password1 != password2:
            raise ValueError("Password don't match.") # not ValidationError!
        return values


# Test with password that don't match:

# In[18]:


try:
    # invalid
    registration = UserRegistration(
        email="user@email.com", 
        password="123456", 
        password_confirmation="1234567"
    )
except ValidationError as e:
    print(e)


# Test with passwords that do match:

# In[19]:


try:
    # valid
    registration = UserRegistration(
        email="user@email.com", 
        password="123456", 
        password_confirmation="123456"
    )
except ValidationError as e:
    print(e)


# The usage of this decorator is similar to the `validator` decorator. The static class
# method is called along with the `values` argument, which is a dictionary containing all
# the fields. Thus, you can retrieve each one of them and implement your logic.
# 
# Once again, Pydantic expects two things for this method. If the values are not valid according to your logic, you should raise a **`ValueError`** error with an explicit error message. Otherwise, you should return a values dictionary that will be assigned to the model. Notice that you could change the values in this dictionary to fit your needs.

# ### Applying validation before Pydantic parsing

# By default, your validators are run after Pydantic has done its parsing work. This means
# that the value you get already conforms to the type of field you specified. If the type is
# incorrect, Pydantic raises an error without calling your validator. Indeed, the following code attempts to perform `int("a3")[1:]`.

# In[20]:


class Test(BaseModel):
    x: int

    @validator("x")
    def x_type(cls, v):
        return v[1:]

try:    
    t = Test(x="a3")
except ValidationError as e:
    print(e)


# To run the validator before Pydantic parses the data, use the setting `pre=True` in the decorator. Here, validation pushes through before type conversion &mdash; essentially performing `int("a3"[1:])` which results to the integer `3`. 

# In[21]:


class Test(BaseModel):
    x: int

    @validator("x", pre=True)
    def x_type(cls, v):
        return v[1:]

try:
    t = Test(x="a3")
    print(t.x)
except ValidationError as e:
    print(e)


# This can be useful if we have preprocessing on a field that will not work if the field has already been converted to its specified type. 

# ## Working with Pydantic objects

# When developing API endpoints with FastAPI, you'll likely get a lot of Pydantic model
# instances to handle. It's then up to you to implement the logic to make a link between those
# objects and your services, such as your database or your **machine learning** (**ML**) model. The trick is to work with Pydantic models as dictionaries when working across different services. Fortunately, Pydantic provides methods to make this very easy. 

# ### Converting an object into a dictionary

# In[22]:


class S(BaseModel):
    x: str
    y: str

class T(BaseModel):
    a: int
    b: int
    c: S
    d: int = -99

t = T(a="0", b="-1", c=S(x="x", y="y"))


# Note that this recursively applies `.dict` to all submodels. Moreover, it provides parameters `include` and `exclude` that expect a set with the keys of the fields you want
# to include or exclude.

# In[23]:


t.dict(include={"a", "b", "c"}, exclude={"b"}) # {"a", "b", "c"} - {"b"}


# We can also apply include or exclude for higher order keys as follows:

# In[24]:


t.dict(include={
        "a": ..., 
        "b": ..., 
        "c": {"x"},
    }
)


# ### Creating an instance from a subclass object

# Since `PostDB` has fields that include the fields of `PostCreate` as a subset, we can create a database object from a post creation object by simply including values from the post creation object's fields. In our case, we do `PostDB(id=new_id, **post_create.dict())` which extracts the key-value pairs to instantiate the fields. 

# ```python
# class DummyDatabase:
#     posts: Dict[int, PostDB] = {}
# 
# db = DummyDatabase()
# 
# @app.post("/posts", status_code=status.HTTP_201_CREATED, response_model=PostPublic)
# async def create(post_create: PostCreate):
#     new_id = max(db.posts.keys() or (0,)) + 1
#     post = PostDB(id=new_id, **post_create.dict())
#     db.posts[new_id] = post
#     return post # PostDB object returned, but JSON response uses fields from PostPublic.
# ```

# Notice also that we set the `response_model` argument on the path operation
# decorator. Basically, it prompts FastAPI to build a **JSON response** with only the fields of
# `PostPublic`, even though we return a PostDB instance at the end of the function.

# ### Updating an instance with a partial one

# In some situations, you'll want to allow **partial updates**. In other words, you'll allow the
# end user to only send the fields they want to change to your API and omit the ones that
# shouldn't change. This is the usual way of implementing a PATCH endpoint.

# To do this, you would first need a special Pydantic model with all the fields marked as
# optional so that no error is raised when a field is missing. Let's see what this looks like
# with our `Post` example, as follows:

# In[25]:


class PostBase(BaseModel):
    title: str
    content: str
    
class PostPartialUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


# We are now able to implement an endpoint that will accept a subset of our `Post` fields.
# Since it's an update, we'll **retrieve** an existing post in the database thanks to its ID. Then,
# we'll have to find a way to only update the fields in the payload and keep the others
# untouched. Fortunately, Pydantic once again has this covered, with handy methods
# and options.

# ```python
# @app.patch("/posts/{id}", response_model=PostPublic)
# async def partial_update(id: int, post_update: PostPartialUpdate):
#     try:
#         post_db = db.posts[id]        
#     except KeyError:
#         raise HTTPException(status.HTTP_404_NOT_FOUND)
# 
#     updated_fields = post_update.dict(exclude_unset=True)
#     updated_post = post_db.copy(update=updated_fields)
# 
#     db.posts[id] = updated_post
#     return updated_post
# ```

# Since we are updating, we first check whether the key exists in the database, otherwise we raise a `KeyError` along with a 404 status code. We update the retrieved `post_db` by applying `.copy(update=updated_fields)` where `updated_fields` is a dictionary that maps fields to new values. The update values are sent via a PATCH request; we use `exclude_unset=True` so that fields not sent in the payload of the PATCH request are ignored in the resulting dictionary. Finally, a JSON response for a public post is returned, and the database object is updated as a side-effect.

# In[26]:


post_retrieved = PostCreate(title="Old Title", content="Old Content.")  # From DB
post_update = PostPartialUpdate(title="New Title")                      # From PATCH request payload 

print("Map of updated fields:", post_update.dict(exclude_unset=True))


# In[27]:


post_retrieved.copy(update=post_update.dict(exclude_unset=True))


# ```{tip}
# You will probably use the `exclude_unset` argument and the `copy` method
# quite often while developing with FastAPI. So be sure to keep them in mind &mdash; they'll make your life easier!
# ```
# 

# 
