from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union, Optional


class TextNode(BaseModel):
    type: Literal['text'] = 'text'
    text: str
    x: int
    y: int
    width: int
    height: int
    rotation: Optional[int] = 0
    opacity: float = 1

    @field_validator('rotation', mode='before')
    @classmethod
    def validate_rotation(cls, v):
        """Convert None to 0 for rotation."""
        return 0 if v is None else v

    # CSS Properties (with aliases for hyphenated JSON keys)
    font_family: str = Field(default='Arial', alias='font-family')
    font_size: int = Field(default=12, alias='font-size')
    color: str = '#000000'
    text_align: str = Field(default='left', alias='text-align')
    font_weight: str = Field(default='normal', alias='font-weight')
    font_style: str = Field(default='normal', alias='font-style')
    font_stretch: str = Field(default='normal', alias='font-stretch')
    text_decoration: str = Field(default='none', alias='text-decoration')
    text_transform: str = Field(default='none', alias='text-transform')
    line_height: float = Field(default=1.2, alias='line-height')

    class Config:
        populate_by_name = True  # Accept both font_family and font-family


class ImageNode(BaseModel):
    type: Literal['image'] = 'image'
    asset_description: str
    filename: Optional[str] = None  # Explicit filename for the asset (e.g., "asset-1.png")
    x: int
    y: int
    width: int
    height: int
    rotation: Optional[int] = 0
    opacity: float = 1

    @field_validator('rotation', mode='before')
    @classmethod
    def validate_rotation(cls, v):
        """Convert None to 0 for rotation."""
        return 0 if v is None else v


class SVGNode(BaseModel):
    type: Literal['svg'] = 'svg'
    svg_description: str  # Description of the vector graphic to generate
    filename: Optional[str] = None  # Explicit filename for the SVG (e.g., "svg-1.svg")
    x: int
    y: int
    width: int
    height: int
    rotation: Optional[int] = 0
    opacity: float = 1

    @field_validator('rotation', mode='before')
    @classmethod
    def validate_rotation(cls, v):
        """Convert None to 0 for rotation."""
        return 0 if v is None else v


Node = Union[TextNode, ImageNode, SVGNode]


class Spec(BaseModel):
    canvas_width: int
    canvas_height: int
    background_color: str
    has_background_image: bool
    background_image_description: Optional[str] = None
    nodes: list[Node] = Field(default_factory=list)