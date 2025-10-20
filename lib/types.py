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

    # CSS Properties
    font_family: str = 'Arial'
    font_size: int = 12
    color: str = '#000000'
    text_align: str = 'left'
    font_weight: str = 'normal'
    font_style: str = 'normal'
    text_decoration: str = 'none'
    text_transform: str = 'none'


class ImageNode(BaseModel):
    type: Literal['image'] = 'image'
    asset_description: str
    x: int
    y: int
    width: int
    height: int
    rotation: int = 0


Node = Union[TextNode, ImageNode]


class Spec(BaseModel):
    background_color: str
    has_background_image: bool
    background_image_description: Optional[str] = None
    nodes: list[Node] = Field(default_factory=list)