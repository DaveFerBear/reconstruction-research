from pydantic import BaseModel, Field
from typing import Literal, Union, Optional


class TextNode(BaseModel):
    type: Literal['text'] = 'text'
    text: str
    x: int
    y: int
    width: int
    height: int
    rotation: int = 0
    opacity: float = 1

    # CSS Properties (with aliases for hyphenated JSON keys)
    font_family: str = Field(default='Arial', alias='font-family')
    font_size: int = Field(default=12, alias='font-size')
    color: str = '#000000'
    text_align: str = Field(default='left', alias='text-align')
    font_weight: str = Field(default='normal', alias='font-weight')
    font_style: str = Field(default='normal', alias='font-style')
    text_decoration: str = Field(default='none', alias='text-decoration')
    text_transform: str = Field(default='none', alias='text-transform')

    class Config:
        populate_by_name = True  # Accept both font_family and font-family


class ImageNode(BaseModel):
    type: Literal['image'] = 'image'
    asset_description: str
    x: int
    y: int
    width: int
    height: int
    rotation: int = 0
    opacity: float = 1


Node = Union[TextNode, ImageNode]


class Spec(BaseModel):
    canvas_width: int
    canvas_height: int
    background_color: str
    has_background_image: bool
    background_image_description: Optional[str] = None
    nodes: list[Node] = Field(default_factory=list)