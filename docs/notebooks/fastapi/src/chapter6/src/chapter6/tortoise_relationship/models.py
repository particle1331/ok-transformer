from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, validator
from tortoise.models import Model
from tortoise import fields


#### Pydantic models #### 

class PostBase(BaseModel):
    title: str
    content: str
    publication_date: datetime = Field(default_factory=datetime.now)

    class Config:
        orm_mode = True


class PostPartialUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


class PostCreate(PostBase):
    pass


class PostDB(PostBase):
    id: int


#### Database entities ####

class PostTortoise(Model):
    id = fields.IntField(pk=True, generated=True)
    publication_date = fields.DatetimeField(null=False)
    title = fields.CharField(max_length=255, null=False)
    content = fields.TextField(null=False)

    class Meta:
        table = "posts"
