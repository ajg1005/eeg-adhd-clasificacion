from pydantic import BaseModel


class FlexibleSchema(BaseModel):
    class Config:
        extra = "allow"
