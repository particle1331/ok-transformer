#!/usr/bin/env python
# coding: utf-8

# # Databases and Asynchronous ORMs

# ```{admonition} Attribution
# This notebook follows Chapter 6: *Databases and Asynchronous ORMs* of {cite}`Voron2021`. Source files for running the background local servers can be found [here](https://github.com/particle1331/machine-learning/tree/master/docs/notebooks/fastapi/src/chapter6).
# ```

# The main goal of a REST API is, of course, to read and write data. So far, we've solely
# worked with the tools given by Python and FastAPI, allowing us to build reliable
# endpoints to process and answer requests. However, we haven't been able to effectively
# retrieve and persist that information: we missed a **database**. 
# 
# In this notebook we will deal with interacting with databases and related libraries inside FastAPI. Note that FastAPI is completely agnostic regarding databases and leaves integration of any system to the developer. We will review three different approaches to integrate a database:
# (1) using **basic SQL queries**, and (2) using **Object-Relational Mapping** (**ORM**).

# ## An overview of relational databases

# The role of a database is to store data in a structured way, preserve the integrity of the
# data, and offer a query language that enables you to retrieve this data when an application
# needs it. Relational databases implement the relational model: each entity, or object, of the
# application is stored in **tables**. Each table has several **columns** containing attributes of the entity. One of the key points of relational databases is, as their name suggests, relationships. Each
# table can be in relation to others, with rows referring to other rows in other tables. 
# 
# The main motivation behind this is to avoid duplication. Indeed, it wouldn't be very
# efficient to repeat an object's attributes in each related to it. If it needs to be modified
# at some point, we would have to go through each related entity, which is error-prone and puts data
# consistency at risk. This is why we prefer to references to entities using unique identifiers. 
# 
# To do this, each row in a relational database has an identifier, called a **primary key**. This is
# unique in the table and will allow you to uniquely identify this row. Therefore, it's possible
# to use this key in another table to reference it. We call it a **foreign key**: the key is foreign in
# the sense that it refers to another table. Relational databases are designed to perform **join queries** efficiently, which will return all the relevant records
# based on the foreign keys. However, those operations can become expensive if the schema is more complex. This is why it's important to carefully design a relational schema and its queries.

# ## Communicating with a SQL database with SQLAlchemy

# To begin, we will discuss how to work with a relational database using the SQLAlchemy
# library. Note that we will only consider the core part of the library, which
# only provides the tools to abstract communication with a SQL database. We won't
# consider the ORM part, as, in the next section, we'll focus on another ORM: Tortoise. We will combine SQLAlchemy with the `databases` library by Encode, the same team
# behind Starlette, which provides an asynchronous connection layer for SQLAlchemy:

# ```{figure} ../../img/sqlalch-encode.png
# ---
# name: sqlalch-encode
# ---
# 
# 
# ```

# ### Creating the table schema

# First, you need to define the SQL schema for your tables: the name, the columns, and their
# associated types and properties. In the following example, you can view the definition of the
# `posts` table:

# ```python
# # chapter6/sqlalchemy/models.py
# 
# metadata = sqlalchemy.MetaData()
# posts = sqlalchemy.Table(
#     "posts",
#     metadata,
#     sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
#     sqlalchemy.Column("publication_date", sqlalchemy.DateTime(), nullable=False),
#     sqlalchemy.Column("title", sqlalchemy.String(length=255), nullable=False),
#     sqlalchemy.Column("content", sqlalchemy.Text(), nullable=False),
# )
# ```

# First, we created a `metadata` object. Its role is to keep all the information of a database
# schema together. This is why you should create it only once in your whole project and
# always use the same one throughout.
# 
# Next, we defined a table using the `Table` class. The first argument is the name of the
# table, followed by the metadata object. Then, we list all of the columns that should be
# defined in our table, thanks to the `Column` class. The first argument is the name of the
# column, followed by its [type](https://docs.sqlalchemy.org/en/13/core/type_basics.html#generic-types) and [some options](https://docs.sqlalchemy.org/en/13/core/metadata.html#:~:text=sqlalchemy.schema.Column.__init__). For example, we define
# our `id` column as a primary key with auto-increment, which is quite common in
# a SQL database.
# 
# We will also define the
# corresponding Pydantic models for our post entity in the same file. Since they will be used by FastAPI to
# validate the request payload, they must match the SQL definition to avoid any errors from
# the database when we try to insert a new row later.

# ```python
# # chapter6/sqlalchemy/models.py
# from datetime import datetime
# from typing import Optional
# from pydantic import BaseModel, Field
# 
# 
# class PostBase(BaseModel):
#     title: str
#     content: str
#     publication_date: datetime = Field(default_factory=datetime.now)
# 
# 
# class PostPartialUpdate(BaseModel):
#     title: Optional[str] = None
#     content: Optional[str] = None
# 
# 
# class PostCreate(PostBase):
#     pass
# 
# 
# class PostDB(PostBase):
#     id: int
# ```

# ### Connecting to a database

# #### Setting up connection

# Now that our table is ready, we have to set up the connection between our FastAPI app
# and the database engine.

# ```python
# # chapter6/sqlalchemy/database.py
# from sqlalchemy import create_engine
# from databases import Database
# 
# 
# DATABASE_URL = "sqlite:///chapter6_sqlalchemy.db"
# database = Database(DATABASE_URL)
# sqlalchemy_engine = create_engine(DATABASE_URL)
# 
# 
# def get_database() -> Database:
#     return database
# ```

# Observe that we instantiate a `Database` instance using the database URL. This is the connection layer provided by `databases` that will allow us to perform asynchronous queries. Notice that the standard synchronous connection established in `sqlalchemy_engine` overlaps with `database`. The idea for this is that all our async endpoints will be using `databases`; we will only use `sqlalchemy_engine` once when creating the schema for our database. 
# 
# 
# The function `get_database` will be used as a dependency easily retrieve the database instance in our path operation functions. Setting up a dependency like this instead of directly importing objects will benefit us during automated testing.

# #### Startup and shutdown

# Now, we need to tell FastAPI to open the connection with the database when it starts
# the application and then close it when exiting. FastAPI provides two
# special decorators to perform tasks at startup and shutdown, as you can see in the
# following example:

# ```python
# # chapter6/sqlalchemy/app.py
# 
# app = FastAPI()
# 
# 
# @app.on_event("startup")
# async def startup():
#     await database.connect()
#     metadata.create_all(sqlalchemy_engine)
# 
# 
# @app.on_event("shutdown")
# async def shutdown():
#     await database.disconnect()
# ```

# Additionally, you can see that we call the `create_all` method on the `metadata` object. This is the same `metadata` object we defined in the previous section and that we have
# imported here. The goal of this method is to create the table's schema inside our database.
# Otherwise, our database would be empty and we would not be able to save
# or retrieve data. This method is designed to work with a standard SQLAlchemy engine;
# this is why we instantiated `sqlalchemy_engine` earlier. It has no other use in the application; instead, we will be using `database` which works with our async endpoints. 

# ### Defining dependencies

# We will define two dependencies. Recall that dependency logic are injected in endpoint calls which allows them to use other arguments of the endpoints whose values may also be obtained through a dependency injection. 

# ```python
# # chapter6/sqlalchemy/app.py
# 
# async def get_post_or_404(
#     id: int,
#     database: Database=Depends(get_database)
# ) -> PostDB:
# 
#     select_query = posts.select().where(posts.c.id == id) # overloaded
#     raw_post = await database.fetch_one(select_query)
#     if raw_post is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
#         
#     return PostDB(**raw_post) # raw_post is of type dict
# 
# 
# async def pagination(
#     skip: int=Query(0, ge=0), limit: int=Query(10, ge=0)
# ) -> Tuple[int, int]:
#     
#     capped_limit = min(100, limit)
#     return (skip, capped_limit)
# ```

# Note that chained method calls are automatically transformed into SQL statements in `select_query`. Moreover, the equality operator in `posts.c.id == id` is not merely a Boolean statement but is overloaded to work with SQLAlchemy method calls. A more thorough discussion of chained method calls is presented in the following subsection.

# In[19]:


from chapter6.sqlalchemy.models import posts
posts.c.id == 1


# Finally, observe that we have to use `await` which tells the interpreter that the async method can't continue past that point &mdash; blocked &mdash; until the awaited asynchronous process is finished.

# ### Making insert queries

# Now we're ready to make queries! Let's start with the INSERT queries to create new rows
# in our database. In the following example, you can view an implementation of an endpoint
# to create a new post:

# ```python
# # sqlalchemy/app.py
# 
# @app.post("/posts", response_model=PostDB, status_code=status.HTTP_201_CREATED)
# async def create_post(
#     post: PostCreate, 
#     database: Database=Depends(get_database)
# ) -> PostDB:
#     
#     insert_query = posts.insert().values(post.dict())
#     post_id = await database.execute(insert_query)
#     post_db = await get_post_or_404(post_id, database)
#     
#     return post_db
# ```

# This is a POST endpoint that accepts a payload following the `PostCreate` model. It also injects the database thanks to our `get_database` dependency. Interesting things begin in the body of the function:
# 
# * On the first line, we build our INSERT query. Rather than writing SQL queries by hand, we rely on the SQLAlchemy expression language, which consists of **chained method calls**. Under the hood, SQLAlchemy will build a proper SQL query for our database engine. This is one of the greatest benefits of such libraries: since it produces the SQL query for you, you won't have to modify your source code if you change your database engine.
# 
# +++
# 
# * This query is built directly from the posts object, which is the `Table` instance that
# we defined earlier. By using this object, SQLAlchemy directly understands that the
# query concerns this table and builds the SQL accordingly. We start by calling the `insert` method. Then, we move ahead with the `values`
# method. This simply accepts a dictionary that associates the names of the columns
# with their values. Hence, we just need to call `dict()` on our Pydantic object. This
# is why it's important that our model matches the database schema.
# 
# +++
# 
# * On the second line, we'll actually perform the query. Thanks to `database`, we can
# execute it asynchronously. For an insert query, we'll use the `execute` method,
# which expects the query in an argument.

# An INSERT query will return the primary key (here `id`) of the newly inserted row. This is very important
# because, since we allow the database to automatically increment this identifier, we don't
# know the `id` of our new post beforehand. In fact, we need it to retrieve this new row from the database afterward. By doing this, we ensure we have an exact representation of the current object in the database before
# returning it in the response. For this, we use the `get_post_or_404` dependency defined above.

# In[1]:


get_ipython().system('http POST :8000/posts title="Title #1" content="Content #1"')


# In[2]:


get_ipython().system('http POST :8000/posts title="Title #2" content="Content #2"')


# ### Making select queries

# Now that we can insert new data into our database, we must be able to read it! Typically,
# you'll have two kinds of read endpoints in your API: one to list objects and one to get
# a single object.

# ```python
# @app.get("/posts/{id}", response_model=PostDB)
# async def get_post(post: PostDB=Depends(get_post_or_404)) -> PostDB:
#     return post
# ```

# Recall that `get_post_or_404` has a SELECT statement inside it which is why we only need to inject that dependency and return its result. To get the list of all posts, we define the following endpoint which depends on `pagination` for offsets and limits.

# ```python
# # chapter6/sqlalchemy/app.py
# 
# @app.get("/posts")
# async def list_posts(
#     pagination: Tuple[int, int]=Depends(pagination), 
#     database: Database=Depends(get_database)
# ) -> List[PostDB]:
# 
#     skip, limit = pagination
#     select_query = posts.select().offset(skip).limit(limit)
#     rows = await database.fetch_all(select_query)
#     results = [PostDB(**row) for row in rows]
#     
#     return results
# ```

# In[3]:


get_ipython().system('http GET :8000/posts/1')


# In[4]:


get_ipython().system('http GET :8000/posts')


# ### Making update and delete queries

# Finally, let's examine how to update and delete rows in our database. The main
# difference is how you build the query using SQLAlchemy expressions, but the rest of the
# implementation is always the same: (1) **build query**, (2) **execute**, and (3) return the **response**.

# ```python
# # chapter6/sqlalchemy/app.py
# 
# @app.patch("/posts/{id}", response_model=PostDB)
# async def update_post(
#     post_update: PostPartialUpdate,
#     post: PostDB=Depends(get_post_or_404),
#     database: Database=Depends(get_database)
# ) -> PostDB:
# 
#     update_query = (
#         posts.update()
#         .where(posts.c.id == post.id)                 # match post in db
#         .values(post_update.dict(exclude_unset=True)) # set update values
#     )
# 
#     post_id = await database.execute(update_query)
#     post_db = await get_post_or_404(post_id, database)
#     return post_db
# 
# 
# @app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_post(
#     post: PostDB=Depends(get_post_or_404),
#     database: Database=Depends(get_database)
# ):
#     delete_query = posts.delete().where(posts.c.id == post.id) # match post to delete
#     await database.execute(delete_query)
# ```

# Let's test these with the existing posts:

# In[5]:


get_ipython().system('http PATCH :8000/posts/1 title="New Title #1"')


# In[6]:


get_ipython().system('http DELETE :8000/posts/2')


# Select all posts:

# In[7]:


get_ipython().system('http :8000/posts')


# ### Adding relationships

# Quite often, you'll need to create entities that are linked to others.
# For example, comments are linked to the post they relate to. In this
# section, we'll examine how you can set up such relationships with SQLAlchemy. First, we need to define the table for the comments, which has a foreign key toward the
# posts table. You can view its definition in the following example:

# ```python
# # chapter6/sqlalchemy_relationship/models.py
# 
# comments = sqlalchemy.Table(
#     "comments",
#     metadata,
#     sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
#     sqlalchemy.Column("post_id", sqlalchemy.ForeignKey("posts.id", ondelete="CASCADE"), nullable=False),
#     sqlalchemy.Column("publication_date", sqlalchemy.DateTime, nullable=False),
#     sqlalchemy.Column("content", sqlalchemy.Text, nullable=False)
# )
# ```

# The important point here is the `post_id` column, which is of the `ForeignKey` type.
# This is a special type that tells SQLAlchemy to automatically handle the type of the
# column and the associated constraint. We simply have to give the table and column name
# it refers to. Note that we can also specify the ON DELETE action. Here we choose `"CASCADE"` which means that all comments will be deleted once the parent post is deleted.
# 
# Here we define Pydantic models for the comments which are quite
# straightforward. Note that `post_id` is specified in the comments base class, along with publication date and content. However, we want to highlight a new model we created `PostPublic` for the posts. This is shown in the following example:

# ```python
# # chapter6/sqlalchemy_relationship/models.py
# 
# class CommentBase(BaseModel):
#     post_id: int
#     content: str
#     publication_date: datetime = Field(default_factory=datetime.now)
# 
# 
# class CommentDB(CommentBase):
#     id: int
# 
# 
# class CommentCreate(CommentBase):
#     pass
# 
# 
# class PostPublic(PostDB):
#     comments: List[CommentDB]
# ```

# In a REST API, there are some cases where it makes sense to automatically
# retrieve the associated objects of an entity. Here, it'll be convenient to get the comments
# of a post in a single request. We'll use this model when getting a single post to serialize the
# comments along with the post data.

# ```python
# # chapter6/sqlalchemy_relationship/app.py
# 
# @app.post("/comments", response_model=CommentDB, status_code=status.HTTP_201_CREATED)
# async def create_comment(
#     comment: CommentCreate,
#     database: Database=Depends(get_database),
# ) -> CommentDB:
# 
#     # First, we must make sure posts exist before making comment
#     select_post_query = posts.select().where(posts.c.id == comment.post_id)
#     post = await database.fetch_one(select_post_query)
# 
#     if post is None:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, 
#             detail=f"Post {comment.post_id} does not exist."
#         )
#     
#     # Now, create comment in the database
#     insert_query = comments.insert().values(comment.dict())
#     comment_id = await database.execute(insert_query)
# 
#     # Build the endpoint response
#     select_query = comments.select().where(comments.c.id == comment_id)
#     raw_comment = await database.fetch_one(select_query)
#     return CommentDB(**raw_comment)
# ```

# Earlier, we mentioned that we wanted to retrieve a post and its comments at the same
# time. To do this, we'll have to make a second query to retrieve the comments and then
# merge all the data together in a `PostPublic` instance. We added this logic in the
# `get_post_or_404` dependency, as you can see in the following example. Note we change the output type to `PostPublic` from `PostDB`. Note that this one change influences all other GET endpoints, though we still have to change the responses of the other endpoints to `PostPublic` to show the comments list. 

# ```python
# # chapter6/sqlalchemy_relationship/app.py
# 
# async def get_post_or_404(
#     id: int,
#     database: Database=Depends(get_database) 
# ) -> PostPublic:
# 
#     select_query = posts.select().where(posts.c.id == id) # overloaded
#     raw_post = await database.fetch_one(select_query)
#     if raw_post is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
# 
#     # Here we add querying all comments for the post
#     select_post_comments_query = comments.select().where(comments.c.post_id == id)
#     raw_comments = await database.fetch_all(select_post_comments_query) # list
#     comments_list = [CommentDB(**comment) for comment in raw_comments]
# 
#     return PostPublic(**raw_post, comments=comments_list)
# ```

# In[8]:


get_ipython().system('http POST :8000/posts title="Title #1" content="Content #1"')


# In[9]:


get_ipython().system('http POST :8000/comments post_id=1 content="Comment for Post #1."')


# In[10]:


get_ipython().system('http GET :8000/posts/1')


# ### Setting up a database migration system with Alembic

# When developing an application, you'll likely make changes to your database schema to
# add new tables, add new columns, or modify existing ones. Of course, if your application
# is already in production, you don't want to erase all your data to recreate the schema from
# scratch: you want them to be migrated to the new schema. Tools for this task have been
# developed, and in this section, we'll learn how to set up Alembic, from the creators of
# SQLAlchemy.
# 

# When starting a new project, the first thing to do is to initialize
# the migration environment, which includes a set of files and directories where Alembic
# will store its configuration and migration files. At the root of your project, run the
# following command:

# ```
# $ alembic init alembic
# ```

# Note that it created an `alembic.ini` file, which contains all the
# configuration options of Alembic. We'll review two important settings of this file:
# `script_location` and `sqlalchemy.url`.
# 
# ```
# # chapter6/sqlalchemy_relationship/alembic.ini
# 
# script_location = alembic
# sqlalchemy.url = sqlite:///chapter6_sqlalchemy.db
# ```

# Next, we'll focus on the `env.py` file. This is a Python script containing all the logic
# executed by Alembic to initialize the migration engine and execute the migrations. Being
# a Python script allows us to finely customize the execution of Alembic. For the time being,
# we'll keep the default one except for one thing: we'll import our metadata object.

# ```python
# # chapter6/sqlalchemy_relationship/alembic/env.py
# 
# target_metadata = metadata
# ```

# We provide the metadata object, which contains all table definitions, so the migration system will be able to automatically generate the migration scripts just
# by looking at your schema! This way, you won't have to write them from scratch.
# When you have made changes to your database schema, you can run the following
# command to generate a new migration script:
# 
# ```
# $ alembic revision --autogenerate -m "Initial migration"
# ```

# ```{margin}
# ⚠️ You should be extremely careful when you run such commands on
# your database, especially on a production one. Very bad things can happen if you make
# a mistake, and you can lose precious data. Always test your migrations in
# a test environment and have fresh and working backups before running them on your
# production database.
# ```
# 
# 
# It'll create a new script in the version's directory with the commands reflecting your schema
# changes. Here, we have the required operations to create our posts and comments table, with
# all of their columns and constraints. Notice that we have two functions: `upgrade` and
# `downgrade`. The first one is used to apply the migration and the second one is used to roll
# it back. This is very important because if something goes wrong during the migration,
# or if you need to revert to an older version of your application, you'll be able to do so
# without breaking your data.
# 
# Finally, you can apply the migrations to your database using the following command:
# 
# ```
# $ alembic upgrade head
# ```
# 
# This will run all the migrations that have not yet been applied to your database until the
# latest. It's interesting to know that, in the process, Alembic creates a table in your database
# so that it can remember all the migrations it has applied: this is how it detects which
# scripts to run.

# ## Communicating with a SQL database with Tortoise ORM

# When dealing with relational databases, you might wish to abstract away the SQL
# concepts and only deal with proper objects from the programming language. That's
# the main motivation behind ORM tools. In this section, we'll examine how to work with Tortoise ORM, which is a modern and asynchronous ORM that fits nicely within a FastAPI project. 

# ### Creating database models

# The first step is to create the Tortoise model for your entity. This is a Python class whose
# attributes represent the columns of your table. This class will provide you static methods
# in which to perform queries, such as retrieving or creating data. Moreover, the actual
# entities of your database will be instances of this class, giving you access to its data like any
# other object. Under the hood, the role of Tortoise is to make the link between this Python
# object and the row in the database. Let's take a look at the definition of our blog post
# model in the following example:

# ```{margin}
# You can find the complete list of [field classes](https://tortoise-orm.readthedocs.io/en/latest/fields.html) in the official docs.
# ```

# ```python
# # chapter6/tortoise/models.py
# 
# class PostTortoise(Model):
#     id = fields.IntField(pk=True, generated=True)
#     publication_date = fields.DatetimeField(null=False)
#     title = fields.CharField(max_length=255, null=False)
#     content = fields.TextField(null=False)
# 
#     class Meta:
#         table = "posts"
# ```

# Notice that we also have a sub-class called `Meta`, which allows us to set some options for
# our table. Here, the table attribute allows us to control the name of the table.
# 
# We also define
# the corresponding Pydantic models for our post entity. They will be used by FastAPI to
# perform data validation and serialization. As you can see in the following example,
# we added a `Config` sub-class and set an attribute called `orm_mode`. This option will allow us to transform an ORM object instance into a Pydantic object
# instance. This is essential because FastAPI is designed to work with Pydantic models,
# not ORM models.

# ```python
# # chapter6/tortoise/models.py
# 
# class PostBase(BaseModel):
#     title: str
#     content: str
#     publication_date: datetime = Field(default_factory=datetime.now)
# 
#     class Config:
#         orm_mode = True
# 
# 
# class PostPartialUpdate(BaseModel):
#     title: Optional[str] = None
#     content: Optional[str] = None
# 
# 
# class PostCreate(PostBase):
#     pass
# 
# 
# class PostDB(PostBase):
#     id: int
# ```

# Here, we hit what is maybe the most confusing part about working with FastAPI and an
# ORM: we'll have to work with both ORM objects and Pydantic models and find ways to
# transform them back and forth.

# ### Setting up the Tortoise engine

# Now that we have our model ready, we have to configure the Tortoise engine to set the
# database connection string and the location of our models. To do this, Tortoise comes
# with a utility function for FastAPI that does all the required tasks for you. In particular,
# it automatically adds event handlers to open and close the connection at startup and
# shutdown; this is something we had to do by hand with SQLAlchemy.

# ```python
# # chapter6/tortoise/app.py
# 
# TORTOISE_ORM = {
#     "connections": {"default": "sqlite://chapter6_tortoise.db"},
#     "apps": {
#         "models": {
#             "models": ["chapter6.tortoise.models"],
#             "default_connection": "default",
#         },
#     },
# }
# 
# register_tortoise(
#     app,
#     config=TORTOISE_ORM,
#     generate_schemas=True,
#     add_exception_handlers=True,
# )
# ```

# As you can see, we put the main configuration options in a variable named `TORTOISE_ORM`. Let's review its different fields:
# 
# *  The connections key contains a dictionary associating a database alias to
# a connection string, which gives access to your database. It follows the standard
# convention, as explained in [the docs](https://tortoise-orm.readthedocs.io/en/latest/databases.html?highlight=db_url#db-url). In most projects, you'll probably have one database named default, but it allows
# you to set several databases if needed.
# 
# +++
# 
# * In the apps key, you'll be able to declare all your modules containing your Tortoise models. The first key just below apps, that is, `models`, will be the prefix with which you'll be able to refer to the associated models. Note that this name is arbitrarily chosen, but is especially important when defining foreign keys. For example, with this configuration, our `PostTortoise` model can be referred to by the name `models.PostTortoise`. It's not the actual path to your module. Underneath it, you have to list all the modules containing your models.
# Additionally, we set the corresponding database connection with the alias
# we defined earlier.

# Then, we call the `register_tortoise` function that'll take care of setting up Tortoise
# for FastAPI. Setting `generate_schemas` to `True` will automatically create the table's schema
# in the database. Otherwise, our database will be empty and we won't be able to
# insert any rows. While this is useful for testing purposes, in a real-world application, you should
# have a proper migration system whose role is to make sure your database schema
# is in sync. We do this below.
# 
# 

# ### Creating objects

# Let's start by inserting new objects inside our database. The main challenge is to transform
# the Tortoise object instance into a Pydantic model.

# ```python
# # chapter6/tortoise/app.py
# 
# @app.post("/posts", response_model=PostDB, status_code=status.HTTP_201_CREATED)
# async def create_post(post: PostCreate) -> PostDB:
#     post_tortoise = await PostTortoise.create(**post.dict())
#     return PostDB.from_orm(post_tortoise) # Recall from_orm = True
# ```

# You can see that the implementation is quite straightforward compared to using SQL. Moreover, this operation is natively asynchronous!

# ### Getting and filtering objects

# Usually, a REST API provides two types of endpoints to read data: one to list objects
# and one to get a specific object. This is exactly what we'll review next! In the following example, you can see how we implemented the endpoint to list objects:

# ```python
# # chapter6/tortoise/app.py
# 
# async def get_post_or_404(id: int) -> PostTortoise:
#     return await PostTortoise.get(id=id)
# 
# 
# @app.get("/posts")
# async def list_posts(pagination: Tuple[int, int]=Depends(pagination)) -> List[PostDB]:
#     skip, limit = pagination
#     posts = await PostTortoise.all().offset(skip).limit(limit)
#     results = [PostDB.from_orm(post) for post in posts]
#     return results
# 
# 
# @app.get("/posts/{id}", response_model=PostDB)
# async def get_post(post: PostTortoise=Depends(get_post_or_404)) -> PostDB:
#     return PostDB.from_orm(post)
# ```

# The dependency takes the `id` in the path parameter and retrieve a single object from the database that corresponds to this identifier. The `get` method is a convenient shortcut for this: if no matching record is found, it raises the `DoesNotExist` exception. If there is more than one matching record, it raises
# `MultipleObjectsReturned`. Remember that we set up Tortoise with
# the `add_exception_handlers` option: under the hood, it adds a global handler that
# automatically catches `DoesNotExist` and builds a proper 404 error in the console.

# ### Updating and deleting objects

# For updating and deleting objects the logic is always
# the same; we just have to adapt the methods we call on our Tortoise object.

# ```python
# # chapter6/tortoise/app.py
# 
# @app.patch("/posts/{id}", response_model=PostDB)
# async def update_post(
#     post_update: PostPartialUpdate,
#     post: PostTortoise=Depends(get_post_or_404)
# ) -> PostDB:
# 
#     post.update_from_dict(post_update.dict(exclude_unset=True))
#     await post.save()
#     return PostDB.from_orm(post)
# 
# 
# @app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_post(post: PostTortoise=Depends(get_post_or_404)):
#     await post.delete()
# ```

# That's almost it for the basics of working with Tortoise ORM. We only covered
# the most basic queries, but you can do far more complex things. You can find a thorough
# overview of the query language in the official documentation [here](https://tortoise-orm.readthedocs.io/en/latest/query.html#query-api).

# In[13]:


get_ipython().system('http POST :8000/posts title="Post #1" content="Content #1"')


# In[14]:


get_ipython().system('http POST :8000/posts title="Post #2" content="Content #2"')


# In[15]:


get_ipython().system('http PATCH :8000/posts/1 content="New Content #1"')


# In[16]:


get_ipython().system('http :8000/posts')


# In[17]:


get_ipython().system('http DELETE :8000/posts/2')


# In[18]:


get_ipython().system('http :8000/posts/1')


# In[19]:


get_ipython().system('http :8000/posts/2')


# ### Adding relationships

# Once again, we'll examine how
# to implement comments that are linked to posts. One of the main tasks of Tortoise, and
# ORM in general, is to ease the process of working with related entities, by automatically
# making the required JOIN queries and instantiating sub-objects.

# ```python
# # chapter6/tortoise_relationship/models.py
# 
# class CommentTortoise(Model):
#     id = fields.IntField(pk=True, generated=True)
#     post = fields.ForeignKeyField(
#         "models.PostTortoise", 
#         related_name="comments", null=False
#     )
#     publication_date = fields.DatetimeField(null=False)
#     content = fields.TextField(null=False)
# 
#     class Meta:
#         table = "comments"
# ```

# The main point of interest here is the `post` field, which is purposely defined as a foreign
# key. The first argument is the reference to the associated model. Notice that we use the
# `models` prefix; this is the same one we defined in the Tortoise configuration that we saw
# earlier. Additionally, we set the `related_name`. This is a typical and convenient feature
# of ORM. By doing this, we'll be able to get all the comments of a given post simply by
# accessing its comments property. The action of querying the related comments, therefore,
# becomes completely *implicit*.
# 
# We also define the corresponding Pydantic models:

# ```python
# # chapter6/tortoise_relationship/models.py
# 
# class CommentBase(BaseModel):
#     post_id: int
#     publication_date: datetime = Field(default_factory=datetime.now)
#     content: str
# 
#     class Config:
#         orm_mode = True
# 
# 
# class CommentCreate(CommentBase):
#     pass
# 
# 
# class CommentDB(CommentBase):
#     id: int
# ```

# Here, you can see that we have defined a `post_id` attribute. This attribute will be used in
# the request payload to set the post that we want to attach this new comment to. When you
# provide this attribute to Tortoise, it automatically understands that you are referring to the
# identifier of the foreign key field, called `post`.

# In a REST API, sometimes, it makes sense to automatically retrieve the associated objects
# of an entity in one request. Here, we'll ensure that the comments of a post are returned in
# the form of a list along with the post data. To do this, we introduce a new Pydantic model,
# `PostPublic`.

# ```python
# # chapter6/tortoise_relationship/models.py
# 
# class PostPublic(PostDB):
#     comments: List[CommentDB]
# 
#     @validator("comments", pre=True)
#     def fetch_comments(cls, v):
#         return list(v) # preprocess: convert to list
# ```

# Earlier, we mentioned that thanks to Tortoise, we can retrieve the comments of a post by simply
# doing `post.comments`. This is convenient, but this attribute is not directly a list of data:
# it's a query set object. If we don't do anything, then, when we try to transform the ORM
# object into a `PostPublic`, Pydantic will try to parse this query set and fail. However,
# calling list on this query set forces it to output the data. That is the purpose of this
# validator. Notice that we set it with `pre=True` to make sure it's called before the built-in
# Pydantic validation.

# ```python
# # chapter6/tortoise_relationship/app.py
# 
# @app.post("/comments", response_model=CommentDB, status_code=status.HTTP_201_CREATED)
# async def create_comment(comment: CommentBase) -> CommentDB:
#     
#     # First check if post exists
#     try:
#         await PostTortoise.get(id=comment.post_id)
#     except:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Post {id} does not exist"
#         )
#         
#     # Create comment
#     comment_tortoise = await CommentTortoise.create(**comment.dict())
#     return CommentDB.from_orm(comment_tortoise)
# ```

# Most of the logic is very similar to the create post endpoint. The main difference is that
# we first check for the existence of the post before proceeding with the comment creation.
# Indeed, we want to avoid the foreign key constraint error that could occur at the database
# level and show a clear and helpful error message to the end user instead.
# 
# As we mentioned earlier, our objective is to output the comments when retrieving a single
# post. To do this, we made a small change to the `get_post_or_404`. We also make a few changes on the endpoints for selecting all posts and selecting a single post.

# ```python
# # chapter6/tortoise_relationship/app.py
# 
# async def get_post_or_404(id: int) -> PostTortoise:
#     try:
#         return await PostTortoise.get(id=id).prefetch_related("comments")
#     except DoesNotExist:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
# 
# 
# @app.get("/posts")
# async def list_posts(pagination: Tuple[int, int] = Depends(pagination)) -> List[PostPublic]:
#     skip, limit = pagination
#     posts = await PostTortoise.all().prefetch_related("comments").offset(skip).limit(limit)
#     results = [PostPublic.from_orm(post) for post in posts]
#     return results
# 
# 
# @app.get("/posts/{id}", response_model=PostPublic)
# async def get_post(post: PostTortoise = Depends(get_post_or_404)) -> PostPublic:
#     return PostPublic.from_orm(post)
#     
# ```

# For fun, we list all posts along with all its comments. Testing everything:

# In[54]:


get_ipython().system('http POST :8000/posts title="Post #1" content="Hello #1"')


# In[55]:


get_ipython().system('http POST :8000/posts title="Post #2" content="Hello #2"')


# In[56]:


get_ipython().system('http PATCH :8000/posts/1 content="Hi #1"')


# In[57]:


get_ipython().system('http DELETE :8000/posts/2')


# In[58]:


get_ipython().system('http POST :8000/comments post_id=1 content="Comment on Post #1."')


# In[59]:


get_ipython().system('http POST :8000/comments post_id=1 content="Another comment on Post #1."')


# In[60]:


get_ipython().system('http :8000/posts/1')


# In[61]:


get_ipython().system('http :8000/posts')


# ### Setting up a database migration system with Aerich

# When you make changes
# to your database schema, you want to migrate your existing data in production in a safe
# and reproducible manner. In this section, we'll demonstrate how to install and configure
# Aerich, which is a database migration tool from the creators of Tortoise.
# 
# ```
# $ pip install aerich
# ```
# 
# The first thing you need to do is declare the Aerich models in your Tortoise configuration.
# Indeed, Aerich stores some migration state information in your database. To do this, add `"models": ["chapter6.tortoise_relationship.models", "aerich.models"]` in `TORTOISE_ORM`. Start initializing:

# ```
# $ aerich init -t chapter6.tortoise_relationship.app.TORTOISE_ORM
# $ aerich init-db
# ```

# ```{margin}
# ⚠️ Aerich migration scripts are not cross-database compatible. This is why you should have the same database engine both in local and in production.
# 
# <br>
# 
# ⚠️ Always review the generated scripts to make sure they correctly reflect your changes and that you don't lose data in the process. Test your migrations in a test environment and have fresh and working
# backups before running them in production.
# ```
# 
# During the life of your project, when you have made changes to your table's schema,
# you'll have to generate new migration scripts to reflect the changes. This is done quite
# easily using the following command. The `--name` option allows you to set a name for your migration. It will automatically generate a new migration file that reflects your changes.
# 
# ```
# aerich migrate --name added_new_tables
# ```
# 
# To apply the migrations to your database, simply run the following command:
# 
# ```
# aerich upgrade
# ```
# 

# ## Conclusion
# 
# As you know, databases are an essential part of every system;
# they allow you to save data in a structured way and retrieve it precisely and reliably
# thanks to powerful query languages. You are now able to leverage their power in FastAPI for relational databases. Additionally,
# you've seen the differences between working with and without an ORM to manage
# relational databases, and you have also learned about the importance of a good migration
# system when working with such databases.
# Serious things can now happen: users can send and retrieve data to and from your system.
# However, this poses a new challenge to tackle. This data needs to be protected so that
# it can remain private and secure. This is exactly what we'll discuss in the next notebook:
# how to authenticate users and set up FastAPI for maximum security.
