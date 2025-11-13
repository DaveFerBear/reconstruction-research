"""Font loading utilities for local and web fonts."""

from pathlib import Path
from typing import Set


# Local fonts available in the fonts/ directory
LOCAL_FONTS = {
    'Tusker Grotesk', 'Saveur Sans Round', 'Eurotype BKL', 'Extenda',
    'Bobby Jones Soft', 'Bobby Jones Soft Outline', 'Bobby Rough Soft',
    'Bobby Rough Soft Outline', 'Ashing', 'Bondjlo', 'Dodo',
    'English 111 Presto', 'Faylake', 'Felt Tip', 'Fontuna Stencil',
    'Merisca', 'Natalic'
}

# System fonts that don't need loading
SYSTEM_FONTS = {
    'Arial', 'Helvetica', 'Times New Roman', 'Georgia',
    'Courier New', 'Verdana', 'Times', 'Courier', 'serif', 'sans-serif'
}


def get_fonts_directory() -> Path:
    """Get the path to the fonts directory."""
    lib_dir = Path(__file__).parent
    return lib_dir.parent / 'fonts'


def generate_font_face_css(font_name: str, fonts_dir: Path) -> str:
    """Generate @font-face CSS for a single local font.

    Args:
        font_name: Name of the font
        fonts_dir: Path to the fonts directory

    Returns:
        CSS string with @font-face declarations
    """
    css_parts = []

    if font_name == 'Tusker Grotesk':
        # Tusker has multiple width variants
        variants = [
            ('1500Medium', 500, '50%'),
            ('2500Medium', 500, '62.5%'),
            ('3500Medium', 500, '75%'),
            ('4500Medium', 500, '87.5%'),
            ('5500Medium', 500, '100%'),
            ('6500Medium', 500, '112.5%'),
            ('5600Semibold', 600, '100%'),
            ('5700Bold', 700, '100%'),
            ('5800Super', 800, '100%'),
        ]
        for variant, weight, stretch in variants:
            font_path = fonts_dir / 'Tusker' / f'TuskerGrotesk-{variant}.ttf'
            if font_path.exists():
                css_parts.append(f"""@font-face {{
    font-family: 'Tusker Grotesk';
    src: url('file://{font_path.resolve()}') format('truetype');
    font-weight: {weight};
    font-stretch: {stretch};
}}""")

    elif font_name == 'Saveur Sans Round':
        for variant, weight in [('Light', 300), ('Regular', 400), ('Semi-bold', 600), ('Bold', 700)]:
            font_path = fonts_dir / 'Saveur Sans Round' / f'SaveurSansRound-{variant}.ttf'
            if font_path.exists():
                css_parts.append(f"""@font-face {{
    font-family: 'Saveur Sans Round';
    src: url('file://{font_path.resolve()}') format('truetype');
    font-weight: {weight};
}}""")

    elif font_name == 'Eurotype BKL':
        variants = [
            ('EurotypoBKL.otf', 400, 'normal'),
            ('EurotypoBKL-Bold.otf', 700, 'normal'),
            ('EurotypoBKL-Heavy.otf', 900, 'normal'),
            ('EurotypoBKL-Italic.otf', 400, 'italic'),
            ('EurotypoBKL-BoldItalic.otf', 700, 'italic'),
        ]
        for filename, weight, style in variants:
            font_path = fonts_dir / 'Eurotype BKL' / filename
            if font_path.exists():
                css_parts.append(f"""@font-face {{
    font-family: 'Eurotype BKL';
    src: url('file://{font_path.resolve()}') format('opentype');
    font-weight: {weight};
    font-style: {style};
}}""")

    elif font_name == 'Extenda':
        for variant, stretch in [('40-Hecto', '75%'), ('80-Peta', '125%')]:
            font_path = fonts_dir / 'Extenda' / f'Extenda-{variant}-trial.ttf'
            if font_path.exists():
                css_parts.append(f"""@font-face {{
    font-family: 'Extenda';
    src: url('file://{font_path.resolve()}') format('truetype');
    font-weight: 400;
    font-stretch: {stretch};
}}""")

    # Single-style fonts
    elif font_name in ['Ashing', 'Bondjlo', 'Dodo', 'English 111 Presto', 'Faylake',
                      'Felt Tip', 'Fontuna Stencil', 'Merisca', 'Natalic']:
        font_mapping = {
            'Ashing': 'Ashing Regular/ahsing-regular.otf',
            'Bondjlo': 'Bondjlo/Bondjlo.ttf',
            'Dodo': 'Dodo Regular/dodo-regular.ttf',
            'English 111 Presto': 'English 111 Presto Regular/English 111 Presto Regular.otf',
            'Faylake': 'Faylake/Faylake.ttf',
            'Felt Tip': 'Felt Tip Regular/Felt Tip Regular.ttf',
            'Fontuna Stencil': 'Fontuna Stencil/fontuna-stencil.otf',
            'Merisca': 'Merisca/merisca.otf',
            'Natalic': 'Natalic/Natalic 2.ttf',
        }

        if font_name in font_mapping:
            font_path = fonts_dir / font_mapping[font_name]
            if font_path.exists():
                fmt = 'opentype' if font_path.suffix == '.otf' else 'truetype'
                css_parts.append(f"""@font-face {{
    font-family: '{font_name}';
    src: url('file://{font_path.resolve()}') format('{fmt}');
    font-weight: 400;
}}""")

    # Bobby Jones variants
    elif font_name in ['Bobby Jones Soft', 'Bobby Jones Soft Outline', 'Bobby Rough Soft', 'Bobby Rough Soft Outline']:
        font_mapping = {
            'Bobby Jones Soft': 'Bobby Jones/Bobby Jones Soft.otf',
            'Bobby Jones Soft Outline': 'Bobby Jones/Bobby Jones Soft Outline.otf',
            'Bobby Rough Soft': 'Bobby Jones/Bobby Rough Soft.ttf',
            'Bobby Rough Soft Outline': 'Bobby Jones/Bobby Rough Soft Outline.ttf',
        }

        font_path = fonts_dir / font_mapping[font_name]
        if font_path.exists():
            fmt = 'opentype' if font_path.suffix == '.otf' else 'truetype'
            css_parts.append(f"""@font-face {{
    font-family: '{font_name}';
    src: url('file://{font_path.resolve()}') format('{fmt}');
    font-weight: 400;
}}""")

    return '\n'.join(css_parts)


def generate_local_fonts_css(fonts_to_load: Set[str]) -> str:
    """Generate @font-face CSS for multiple local fonts.

    Args:
        fonts_to_load: Set of local font names to load

    Returns:
        CSS string with all @font-face declarations
    """
    if not fonts_to_load:
        return ''

    fonts_dir = get_fonts_directory()
    css_parts = []

    for font_name in fonts_to_load:
        if font_name in LOCAL_FONTS:
            font_css = generate_font_face_css(font_name, fonts_dir)
            if font_css:
                css_parts.append(font_css)

    return '\n'.join(css_parts)


def generate_google_fonts_url(fonts_needed: Set[str]) -> str:
    """Generate Google Fonts URL for fonts that aren't local or system fonts.

    Args:
        fonts_needed: Set of all font names needed

    Returns:
        Google Fonts URL string, or empty string if no Google Fonts needed
    """
    fonts_to_load = fonts_needed - SYSTEM_FONTS - LOCAL_FONTS

    if not fonts_to_load:
        return ''

    font_imports = []
    for font in fonts_to_load:
        # Convert font name to Google Fonts format
        font_param = font.replace(' ', '+')
        # Load with multiple weights for flexibility
        font_imports.append(f'family={font_param}:wght@300;400;500;600;700;800;900')

    return f"https://fonts.googleapis.com/css2?{'&'.join(font_imports)}&display=swap"
