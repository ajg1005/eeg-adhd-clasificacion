from pydantic import BaseModel, ConfigDict


class FlexibleSchema(BaseModel):
    model_config = ConfigDict(extra="allow", from_attributes=True)


class OrmSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
