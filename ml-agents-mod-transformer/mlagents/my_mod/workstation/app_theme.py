from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
import time


class Iridescent(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.rose,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Source Sans Pro"),
            fonts.GoogleFont("Quicksand"),
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # body_background_fill_dark ="linear-gradient(45deg, *neutral_900, *primary_900)",
            # body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color_dark="Black",
            button_primary_background_fill="linear-gradient(35deg, *primary_200, #fdfaf7, #F7F9D2, *secondary_300)",
            button_primary_background_fill_dark="linear-gradient(35deg, *primary_200, #fdfaf7, #F7F9D2, *secondary_300)",
            # button_secondary_text_color_dark="Black",
            # button_secondary_background_fill_dark="linear-gradient(90deg, #5fb7cf, *secondary_900)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


class Softy(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.neutral,
        secondary_hue: colors.Color | str = colors.rose,
    ):

        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
        )
        super().set(

        )
